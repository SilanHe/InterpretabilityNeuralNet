#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy.stats import pearsonr,spearmanr
import sys
from os.path import join
sys.path.insert(1, join(sys.path[0], 'train_model'))

from scipy.special import softmax

from train_model import sent_util

import torch
from torchtext import data, datasets
import pandas as pd
import time

# function
def magnitude(l):
	xs = np.array(l)
	xs_abs = np.abs(xs)
	max_index = np.argmax(xs_abs)
	x = xs[max_index]

	return x

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
list_len_sentence = list()

# get sst in tree format
sst_sentences, sst = sent_util.get_sst_PTB("data/trees")
len_sst = len(sst)

start = time.process_time()
for index,tree in enumerate(sst):
	batch = [word.lower() for word in sst_sentences[index]]
	cd,ig,label = sent_util.travelTree_IG_CD(batch, model, inputs, answers, tree, sum) # get ig phrase level scores as well using sum baseline

	list_ig += ig
	list_cd += cd
	list_label += label
	list_len_sentence.append(len(cd))

end = time.process_time()
print("time:",end - start)

list_ig = np.array(list_ig)
list_cd = np.array(list_cd)
list_label = np.array(list_label)

list_sum = list_ig + list_cd # sum of ig + cd baseline
list_reweigh = list() # reweighted baseline using sum
list_softmax_ig = np.zeros(1)
list_softmax_cd = np.zeros(1)

index_list = 0
for len_sentence in list_len_sentence:

	# reweighting baseline, reweigh according to magnitude
	largest_magnitude = magnitude(list_ig[index_list:index_list+len_sentence])

	for index in range(index_list,index_list+len_sentence):
		list_reweigh.append(list_cd[index] * abs(list_ig[index]) / largest_magnitude)

	list_softmax_ig = np.append(list_softmax_ig,softmax(list_ig[index_list:index_list+len_sentence]), axis = 0)
	list_softmax_cd = np.append(list_softmax_cd,softmax(list_cd[index_list:index_list+len_sentence]), axis = 0)

	index_list += len_sentence

list_softmax_sum = list_cd + list_softmax_ig
list_softmax_ig_cd = list_softmax_ig + list_softmax_cd

pearson_corr_cd, _ = pearsonr(list_cd,list_label)
spearman_corr_cd, _ = spearmanr(list_cd,list_label)

pearson_corr_ig, _ = pearsonr(list_ig,list_label)
spearman_corr_ig, _ = spearmanr(list_ig,list_label)

pearson_corr_sum, _ = pearsonr(list_sum,list_label)
spearman_corr_sum, _ = spearmanr(list_sum,list_label) 

pearson_corr_reweigh, _ = pearsonr(list_reweigh,list_label)
spearman_corr_reweigh, _ = spearmanr(list_reweigh,list_label) 

pearson_corr_softmax_sum, _ = pearsonr(list_softmax_sum,list_label)
spearman_corr_softmax_sum, _ = spearmanr(list_softmax_sum,list_label)

pearson_coor_softmax_cd, _ = pearsonr(list_softmax_cd,list_label)
spearman_corr_softmax_cd, _ = spearmanr(list_softmax_cd,list_label) 

pearson_coor_softmax_ig, _ = pearsonr(list_softmax_ig,list_label)
spearman_corr_softmax_ig, _ = spearmanr(list_softmax_ig,list_label)

pearson_corr_softmax, _ = pearsonr(list_softmax_ig_cd,list_label)
spearman_corr_softmax, _ = spearmanr(list_softmax_ig_cd,list_label)

print("______________________________________")
print("CD")
print("Pearson Correlation", pearson_corr_cd)
print("Spearman Correlation", spearman_corr_cd)
print("Covariance", np.cov(list_cd,list_label))

print("______________________________________")
print("IG")
print("Pearson Correlation", pearson_corr_ig)
print("Spearman Correlation", spearman_corr_ig)
print("Covariance", np.cov(list_ig,list_label))

print("______________________________________")
print(" IG + CD")
print("Pearson Correlation", pearson_corr_sum)
print("Spearman Correlation", spearman_corr_sum)
print("Covariance", np.cov(list_sum,list_label))

print("______________________________________")
print("Reweighted CD using IG ratio with max")
print("Pearson Correlation", pearson_corr_reweigh)
print("Spearman Correlation", spearman_corr_reweigh)
print("Covariance", np.cov(list_reweigh,list_label))

print("______________________________________")
print("CD + Softmax IG")
print("Pearson Correlation", pearson_corr_softmax_sum)
print("Spearman Correlation", spearman_corr_softmax_sum)
print("Covariance", np.cov(list_softmax_sum,list_label))

print("______________________________________")
print("Softmax IG")
print("Pearson Correlation", pearson_corr_softmax_ig)
print("Spearman Correlation", spearman_corr_softmax_ig)
print("Covariance", np.cov(list_softmax_ig,list_label))

print("______________________________________")
print("Softmax CD")
print("Pearson Correlation", pearson_corr_softmax_cd)
print("Spearman Correlation", spearman_corr_softmax_cd)
print("Covariance", np.cov(list_softmax_cd,list_label))

print("______________________________________")
print("Softmax CD + Softmax IG")
print("Pearson Correlation", pearson_corr_softmax)
print("Spearman Correlation", spearman_corr_softmax)
print("Covariance", np.cov(list_softmax_ig_cd,list_label))

