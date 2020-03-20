#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy.stats import pearsonr
import sys
from os.path import join
sys.path.insert(1, join(sys.path[0], 'train_model'))

from train_model import sent_util

import torch
from torchtext import data, datasets
import pandas as pd

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


# get list of data with different predicted and true label
list_diff_label = list()
list_cd = np.zeros(1)
list_ig = np.zeros(1)

for ind in range(6919):
	if sent_util.diff_predicted_label(data[ind], model, answers):
		pred, list_scores_cd = sent_util.CD_unigram(data[ind], model, inputs, answers)
		list_cd = np.append(list_cd,list_scores_cd, axis = 0)
		pred, list_scores_ig = sent_util.integrated_gradients_unigram(data[ind], model, inputs, answers)
		list_ig = np.append(list_ig,list_scores_ig, axis = 0)
		list_diff_label.append(ind)

print("______________________________________")
print("Correlation", pearsonr(list_cd,list_ig))
print("Covariance", np.cov(list_cd,list_ig))
print()
print("list of indeces of inputs with differering predicted and true labels:", len(list_diff_label), "/", 6920)
print()
for i in list_diff_label:
	print(i)