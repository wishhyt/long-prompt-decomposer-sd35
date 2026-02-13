#!/usr/bin/env python
# coding=utf-8

import os
import sys

# BOOTSTRAP_PATHS: allow running scripts directly without installing the package.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# system
import argparse
import logging
import math
import os
import re
import json
import random
import shutil
import warnings
import time

# utilities
import torch
import numpy as np
import PIL
import safetensors
import torch
from accelerate import Accelerator
from safetensors.torch import save_file, load_file
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# data
from PIL import Image
from packaging import version
from torchvision import transforms
from datasets import load_dataset
from tqdm.auto import tqdm
from loss import dense_score, clip_score, pick_score

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
PATTERN = re.compile(r'"!|\.|\?|;"')

def main():
    transform = transforms.ToTensor()
    test_dir = "//samples/long_PDD/sd2/dm_bench/llm4gen"
    with open (f"{test_dir}.json", 'r') as f:
        meta = json.load(f)
    with open ("//data-juicer/playground/evaluation/paradiff_5551390_detail_score.json", 'r') as f:
        meta = json.load(f)
    results = 0.0
    count = 0
    # for image in tqdm(os.listdir(test_dir)):
    for res in tqdm(meta):
        image = res['temp_image_name']
        try:
            img = transform(Image.open(os.path.join(test_dir, image))).to('cuda')
        except PIL.UnidentifiedImageError:
            print('PIL cannot identify', image)
            continue
        # img_path = image.split('.')[0]
        # caption = meta[img_path]
        caption = res['prompt_info']['polished_prompt']

        sentence_list, sentence_index = [], []
        sentence_remain = []
        sentence_list_ = re.split(PATTERN, caption)
        sentence_list_ = [sent + '.' for sent in sentence_list_ if len(sent) > 0]
        if len(sentence_list_) == 0:
            sentence_list_ = [caption]
        sentence_index += [0] * len(sentence_list_)
        sentence_list += sentence_list_

        cap_index = sorted(random.sample(range(len(sentence_list_)), min(len(sentence_list_), 4)))  # choose 4 sentences
        cap_ = [sentence_list_[ii].strip() for ii in cap_index]
        sentence_remain.append(' '.join(cap_))
        # score = dense_score(img.unsqueeze(0), sentence_list, sentence_index, do_ortho=False, return_split=False)
        score = pick_score(img.unsqueeze(0), caption, do_ortho=False,)
        # score = clip_score(img.unsqueeze(0), caption, do_ortho=False,)
        results += score
        count += 1
    print(results / count, count)


if __name__ == "__main__":
    main()