#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy.stats import pearsonr,spearmanr
import sys
from os.path import join
sys.path.insert(1, join(sys.path[0], 'train_model'))

from train_model import sent_util

import torch
from torchtext import data, datasets
import pandas as pd
import time

# To train model, first run 'train.py' from train_model dir

# get model path, OS safe

snapshot_dir = 'results_sst/'
snapshot_file = join(snapshot_dir, 'best_snapshot_devacc_79.35779571533203_devloss_0.41613781452178955_iter_9000_model.pt')

# get model
model = sent_util.get_model(snapshot_file)

# get data
inputs, answers, train_iterator, dev_iterator = sent_util.get_sst()

batch_nums = list(range(6920))
data = sent_util.get_batches(batch_nums, train_iterator, dev_iterator)

# setup for data collection
list_cd = np.zeros(1)
list_ig = np.zeros(1)

total_overlap = 0

total_overlap_count = 0
total_overlap_list = list()

# get sst in tree format
sst_sentences, sst = sent_util.get_sst_PTB()
len_sst = len(sst)

start = time.clock()
for index,tree in enumerate(sst):
	batch = [word.lower() for word in sst_sentences[index]]
	sent_util.travelTree(batch, model,tree)
        end = time.clock()
        print("time:",end - start)
        if index > 0:
            break
