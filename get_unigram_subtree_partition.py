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
list_ig = list()
list_cd = list()
list_label = list()

list_ig_unmatched = list()
list_cd_unmatched = list()
list_label_unmatched = list()

# get sst in tree format
sst_sentences, sst = sent_util.get_sst_PTB("data/trees")
len_sst = len(sst)

start = time.process_time()
for index,tree in enumerate(sst):
	batch = [word.lower() for word in sst_sentences[index]]
	ig,cd,label,matches = sent_util.travelTreeUnigram(batch, model, inputs, answers, tree)

	if matches:
		list_ig += ig
		list_cd += cd
		list_label += label
	else:
		list_ig_unmatched += ig
		list_cd_unmatched += cd
		list_label_unmatched += label

end = time.process_time()
print("time:",end - start)

list_ig = np.array(list_ig)
list_cd = np.array(list_cd)
list_label = np.array(list_label)

list_ig_ummatched = np.array(list_ig_unmatched)
list_cd_unmatched = np.array(list_cd_unmatched)
list_label_unmatched = np.array(list_label_unmatched)

pearson_corr_cd, _ = pearsonr(list_cd,list_label)
spearman_corr_cd, _ = spearmanr(list_cd,list_label)

pearson_corr_ig, _ = pearsonr(list_ig,list_label)
spearman_corr_ig, _ = spearmanr(list_ig,list_label)

pearson_corr_cd_unmatched, _ = pearsonr(list_cd_unmatched,list_label_unmatched)
spearman_corr_cd_unmatched, _ = spearmanr(list_cd_unmatched,list_label_unmatched)

pearson_corr_ig_unmatched, _ = pearsonr(list_ig_unmatched,list_label_unmatched)
spearman_corr_ig_unmatched, _ = spearmanr(list_ig_unmatched,list_label_unmatched)

print("______________________________________")
print("Pearson Correlation CD", pearson_corr_cd)
print("Spearman Correlation CD", spearman_corr_cd)
print("Covariance CD", np.cov(list_cd,list_label))

print("______________________________________")
print("Pearson Correlation IG", pearson_corr_ig)
print("Spearman Correlation IG", spearman_corr_ig)
print("Covariance IG", np.cov(list_ig,list_label))

print("--------------------------------------------------------")
print("unmatched labels")

print("______________________________________")
print("Pearson Correlation CD", pearson_corr_cd_unmatched)
print("Spearman Correlation CD", spearman_corr_cd_unmatched)
print("Covariance CD", np.cov(list_cd_unmatched,list_label_unmatched))

print("______________________________________")
print("Pearson Correlation IG", pearson_corr_ig_unmatched)
print("Spearman Correlation IG", spearman_corr_ig_unmatched)
print("Covariance IG", np.cov(list_ig_unmatched,list_label_unmatched))