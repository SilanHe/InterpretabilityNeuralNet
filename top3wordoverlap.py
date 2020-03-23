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


for ind in range(6919):
	
	pred, list_scores_cd = sent_util.CD_unigram(data[ind], model, inputs, answers)
	pred, list_scores_ig = sent_util.integrated_gradients_unigram(data[ind], model, inputs, answers)
	
	list_cd = np.append(list_cd,list_scores_cd, axis = 0)
	list_ig = np.append(list_ig,list_scores_ig, axis = 0)
	if (list_scores_cd.shape[0] > 2 and list_scores_id.shape[0] > 2):
		index_top3_cd = np.argpartition(np.absolute(list_scores_cd), -3)[-3:]
		index_top3_ig = np.argpartition(np.absolute(list_scores_ig), -3)[-3:]
		
		overlap = np.intersect1d(index_top3_cd,index_top3_ig)
		print("-----------------------------")
		print("overlap ^", overlap.shape[0])
		print("-----------------------------")
		total_overlap += overlap.shape[0]

pearson_corr, _ = pearsonr(list_cd,list_ig)
spearman_corr, _ = spearmanr(list_cd,list_ig)

print("______________________________________")
print("Pearson Correlation", pearson_corr)
print("Spearman Correlation", spearman_corr)
print("Covariance", np.cov(list_cd,list_ig))
print("AVG overlap", total_overlap / 6919)