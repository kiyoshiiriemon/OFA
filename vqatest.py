# How to run: PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 vqatest.py *.jpg
# code is from https://www.12-technology.com/2022/03/ofa-image-captioning-vqa-python.html

import torch
import sys
import numpy as np
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from tasks.mm_tasks.refcoco import RefcocoTask
 
from models.ofa import OFAModel
from PIL import Image
import os
import cv2
import numpy
import csv 
tasks.register_task('refcoco', RefcocoTask)
 
# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False
 
# specify some options for evaluation
parser = options.get_generation_parser()
input_args = ["", "--task=refcoco", "--beam=10", "--path=checkpoints/ofa_large_clean.pt", "--bpe-dir=utils/BPE"]
args = options.parse_args_and_arch(parser, input_args)
cfg = convert_namespace_to_omegaconf(args)

task = tasks.setup_task(cfg.task)
models, cfg = checkpoint_utils.load_model_ensemble(
    utils.split_paths(cfg.common_eval.path),
    task=task
)

# GPUに載せる
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# generatorの初期化
generator = task.build_generator(models, cfg.generation)

# Image transform
from torchvision import transforms
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((task.cfg.patch_image_size, task.cfg.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    token_result = []
    bin_result = []
    img_result = []
    for token in x.strip().split():
      if token.startswith('<bin_'):
        bin_result.append(token)
      elif token.startswith('<code_'):
        img_result.append(token)
      else:
        if bpe is not None:
          token = bpe.decode('{}'.format(token))
        if tokenizer is not None:
          token = tokenizer.decode(token)
        if token.startswith(' ') or len(token_result) == 0:
          token_result.append(token.strip())
        else:
          token_result[-1] += token

    return ' '.join(token_result), ' '.join(bin_result), ' '.join(img_result)


def coord2bin(coords, w_resize_ratio, h_resize_ratio):
    coord_list = [float(coord) for coord in coords.strip().split()]
    bin_list = []
    bin_list += ["<bin_{}>".format(int((coord_list[0] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int((coord_list[1] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int((coord_list[2] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int((coord_list[3] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    return ' '.join(bin_list)


def bin2coord(bins, w_resize_ratio, h_resize_ratio):
    bin_list = [int(bin[5:-1]) for bin in bins.strip().split()]
    coord_list = []
    coord_list += [bin_list[0] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[1] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    coord_list += [bin_list[2] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[3] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    return coord_list


def encode_text(text, length=None, append_bos=False, append_eos=False):
    line = [
      task.bpe.encode(' {}'.format(word.strip()))
      if not word.startswith('<code_') and not word.startswith('<bin_') else word
      for word in text.strip().split()
    ]
    line = ' '.join(line)
    s = task.tgt_dict.encode_line(
        line=line,
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

def construct_sample(image: Image, instruction: str):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])

    instruction = encode_text(' {}'.format(instruction.lower().strip()), append_bos=True, append_eos=True).unsqueeze(0)
    instruction_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in instruction])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": instruction,
            "src_lengths": instruction_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        }
    }
    return sample

# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def image_qa(filename, instruction, flag, clearcsv, misscsv):
    image = Image.open(filename)
    cvimage = np.array(image, dtype=np.uint8)
    cvimage = cv2.cvtColor(cvimage, cv2.COLOR_RGB2BGR)
    for instruction in instruction_list:
        # Construct input sample & preprocess for GPU if cuda available
        sample = construct_sample(image, instruction)
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
         
        # Generate result
        with torch.no_grad():
            hypos = task.inference_step(generator, models, sample)
            tokens1, bins1, imgs1 = decode_fn(hypos[0][0]["tokens"], task.tgt_dict, task.bpe, generator)
            tokens2, bins2, imgs2 = decode_fn(hypos[0][1]["tokens"], task.tgt_dict, task.bpe, generator)
            tokens3, bins3, imgs3 = decode_fn(hypos[0][2]["tokens"], task.tgt_dict, task.bpe, generator)
            tokens4, bins4, imgs4 = decode_fn(hypos[0][3]["tokens"], task.tgt_dict, task.bpe, generator)
            tokens5, bins5, imgs5 = decode_fn(hypos[0][4]["tokens"], task.tgt_dict, task.bpe, generator)
         
        # display result
        cv2.imshow('image', cvimage)
        cv2.waitKey(1)
            #print(f'Image {filename}, Instruction: {instruction}')
            #print('OFA\'s Output1: {}, Probs: {}'.format(tokens1, hypos[0][0]["score"].exp().item()))
            #print('OFA\'s Output2: {}, Probs: {}'.format(tokens2, hypos[0][1]["score"].exp().item()))
            #print('OFA\'s Output3: {}, Probs: {}'.format(tokens3, hypos[0][2]["score"].exp().item()))
            #print('OFA\'s Output4: {}, Probs: {}'.format(tokens4, hypos[0][3]["score"].exp().item()))
            #print('OFA\'s Output5: {}, Probs: {}'.format(tokens5, hypos[0][4]["score"].exp().item()))

        print(instruction)
        pred = list([tokens1, tokens2, tokens3, tokens4, tokens5])
        for i in range(5):
            if pred[i] == flag and hypos[0][i]["score"].exp().item() >= 0.5:
                with open(clearcsv, "a", newline="") as f1:
                    writer = csv.writer(f1)
                    writer.writerow([str(instruction), str(filename), str(pred[i])])
                return 1
            else:
                with open(misscsv, "a", newline="") as f2:
                    writer = csv.writer(f2)
                    writer.writerow([str(instruction), str(filename), str(pred[i])])
                return 0


if __name__ == '__main__':
    args = sys.argv
    instruction_list = ["What alphabet is written on the label on the green box?", "What is the biggest alphabet?", "What is the biggest alphabet on the label on the green box?", "What is the biggest alphabet on the white label on the green box?", "What is the biggest alphabet on the label on the box?", "What is the biggest alphabet on the box?", "What is the biggest letter?", "What is the largest alphabet?", "What is the largest alphabet on the label on the green box?", "What is the largest alphabet on the white label on the green box?", "What is the largest alphabet on the label on the box?", "What is the largest alphabet on the box?", "What is the largest letter?", "What is written on the white label?", "Could you tell me the text on the label of the box?", "What does the label on the box say?", "What is the largest character on the label?", "Which character is the biggest on the label?"]
    args.pop(0)
    flag = args.pop(-1)
    score = 0
    result_dir = "result/"+flag
    csv1, csv2 = result_dir+"/clear.csv", result_dir+"/miss.csv"
    
    if os.path.exists(result_dir):
        pass
    else:
        os.mkdir(result_dir)

    if os.path.exists(csv1) and os.path.exists(csv2):
        os.remove(csv1)
        os.remove(csv2)

    for instruction in instruction_list:
        for file in args:
            ans = image_qa(file, instruction, flag, csv1, csv2)
            score += ans
        print("next!!!!")
    
    print(f'正解数：{score}, 正解率：{score}/{len(args)*len(instruction_list)}')
