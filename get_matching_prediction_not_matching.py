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

# setup for data collection: we are partitioning data for both matching and not matching labels between ground truth and prediction
list_ig_matching = list()
list_cd_matching = list()
list_label_matching = list()
list_len_sentence_matching = list()

list_ig_not_matching = list()
list_cd_not_matching = list()
list_label_not_matching = list()
list_len_sentence_not_matching = list()

# get sst in tree format
sst_sentences, sst = sent_util.get_sst_PTB("data/trees")
len_sst = len(sst)

start = time.process_time()
for index,tree in enumerate(sst):
	batch = [word.lower() for word in sst_sentences[index]]
	cd,ig,label,matching = sent_util.travelTree_IG_CD(batch, model, inputs, answers, tree, sum) # get ig phrase level scores as well using sum baseline

	if matching:
		list_ig_matching += ig
		list_cd_matching += cd
		list_label_matching += label
		list_len_sentence_matching.append(len(cd))
	else:
		list_ig_not_matching += ig
		list_cd_not_matching += cd
		list_label_not_matching += label
		list_len_sentence_not_matching.append(len(cd))


end = time.process_time()
print("time:",end - start)

# post processing

# create lists holding data

list_ig_matching = np.array(list_ig_matching)
list_cd_matching = np.array(list_cd_matching)
list_label_matching = np.array(list_label_matching)

list_ig_not_matching = np.array(list_ig_not_matching)
list_cd_not_matching = np.array(list_cd_not_matching)
list_label_not_matching = np.array(list_label_not_matching)

list_sum_matching = list_ig_matching + list_cd_matching # sum of ig + cd baseline
list_sum_not_matching = list_ig_not_matching + list_cd_not_matching
list_reweigh_matching = list() # reweighted baseline using sum
list_reweigh_not_matching = list()

list_softmax_ig_matching = np.array([])
list_softmax_ig_not_matching = np.array([])
list_softmax_cd_matching = np.array([])
list_softmax_cd_not_matching = np.array([])

# iterate through data with matching label
index_list_matching = 0
for len_sentence in list_len_sentence_matching:

	# reweighting baseline, reweigh according to magnitude
	largest_magnitude = magnitude(list_ig_matching[index_list_matching:index_list_matching+len_sentence])

	for index in range(index_list_matching,index_list_matching+len_sentence):
		list_reweigh_matching.append(list_cd_matching[index] * abs(list_ig_matching[index]) / largest_magnitude)

	list_softmax_ig_matching = np.append(list_softmax_ig_matching,softmax(list_ig_matching[index_list_matching:index_list_matching+len_sentence]), axis = 0)
	list_softmax_cd_matching = np.append(list_softmax_cd_matching,softmax(list_cd_matching[index_list_matching:index_list_matching+len_sentence]), axis = 0)

	index_list_matching += len_sentence

list_softmax_sum_matching = list_cd_matching + list_softmax_ig_matching
list_softmax_ig_cd_matching = list_softmax_ig_matching + list_softmax_cd_matching

pearson_corr_cd, _ = pearsonr(list_cd_matching,list_label_matching)
spearman_corr_cd, _ = spearmanr(list_cd_matching,list_label_matching)

pearson_corr_ig, _ = pearsonr(list_ig_matching,list_label_matching)
spearman_corr_ig, _ = spearmanr(list_ig_matching,list_label_matching)

pearson_corr_sum, _ = pearsonr(list_sum_matching,list_label_matching)
spearman_corr_sum, _ = spearmanr(list_sum_matching,list_label_matching) 

pearson_corr_reweigh, _ = pearsonr(list_reweigh_matching,list_label_matching)
spearman_corr_reweigh, _ = spearmanr(list_reweigh_matching,list_label_matching) 

pearson_corr_softmax_sum, _ = pearsonr(list_softmax_sum_matching,list_label_matching)
spearman_corr_softmax_sum, _ = spearmanr(list_softmax_sum_matching,list_label_matching)

pearson_corr_softmax_cd, _ = pearsonr(list_softmax_cd_matching,list_label_matching)
spearman_corr_softmax_cd, _ = spearmanr(list_softmax_cd_matching,list_label_matching) 

pearson_corr_softmax_ig, _ = pearsonr(list_softmax_ig_matching,list_label_matching)
spearman_corr_softmax_ig, _ = spearmanr(list_softmax_ig_matching,list_label_matching)

pearson_corr_softmax, _ = pearsonr(list_softmax_ig_matching_cd,list_label_matching)
spearman_corr_softmax, _ = spearmanr(list_softmax_ig_matching_cd,list_label_matching)

# iterate through data with matching label
index_list_not_matching = 0
for len_sentence in list_len_sentence_not_matching:

	# reweighting baseline, reweigh according to magnitude
	largest_magnitude = magnitude(list_ig_not_matching[index_list_not_matching:index_list_not_matching+len_sentence])

	for index in range(index_list_not_matching,index_list_not_matching+len_sentence):
		list_reweigh_not_matching.append(list_cd_matching[index] * abs(list_ig_not_matching[index]) / largest_magnitude)

	list_softmax_ig_not_matching = np.append(list_softmax_ig_not_matching,softmax(list_ig_not_matching[index_list_not_matching:index_list_not_matching+len_sentence]), axis = 0)
	list_softmax_cd_not_matching = np.append(list_softmax_cd_not_matching,softmax(list_cd_not_matching[index_list_not_matching:index_list_not_matching+len_sentence]), axis = 0)

	index_list_not_matching += len_sentence

list_softmax_sum_not_matching = list_cd_not_matching + list_softmax_ig_not_matching
list_softmax_ig_cd_not_matching = list_softmax_ig_not_matching + list_softmax_cd_not_matching

pearson_corr_cd_not_matching, _ = pearsonr(list_cd_not_matching,list_label_not_matching)
spearman_corr_cd_not_matching, _ = spearmanr(list_cd_not_matching,list_label_not_matching)

pearson_corr_ig_not_matching, _ = pearsonr(list_ig_not_matching,list_label_not_matching)
spearman_corr_ig_not_matching, _ = spearmanr(list_ig_not_matching,list_label_not_matching)

pearson_corr_sum_not_matching, _ = pearsonr(list_sum_not_matching,list_label_not_matching)
spearman_corr_sum_not_matching, _ = spearmanr(list_sum_not_matching,list_label_not_matching) 

pearson_corr_reweigh_not_matching, _ = pearsonr(list_reweigh_not_matching,list_label_not_matching)
spearman_corr_reweigh_not_matching, _ = spearmanr(list_reweigh_not_matching,list_label_not_matching) 

pearson_corr_softmax_sum_not_matching, _ = pearsonr(list_softmax_sum_not_matching,list_label_not_matching)
spearman_corr_softmax_sum_not_matching, _ = spearmanr(list_softmax_sum_not_matching,list_label_not_matching)

pearson_corr_softmax_cd_not_matching, _ = pearsonr(list_softmax_cd_not_matching,list_label_not_matching)
spearman_corr_softmax_cd_not_matching, _ = spearmanr(list_softmax_cd_not_matching,list_label_not_matching) 

pearson_corr_softmax_ig_not_matching, _ = pearsonr(list_softmax_ig_not_matching,list_label_not_matching)
spearman_corr_softmax_ig_not_matching, _ = spearmanr(list_softmax_ig_not_matching,list_label_not_matching)

pearson_corr_softmax_not_matching, _ = pearsonr(list_softmax_ig_cd_not_matching,list_label_not_matching)
spearman_corr_softmax_not_matching, _ = spearmanr(list_softmax_ig_cd_not_matching,list_label_not_matching)



print("______________________________________")
print("CD")
print("Pearson Correlation", pearson_corr_cd)
print("Spearman Correlation", spearman_corr_cd)
print("Covariance", np.cov(list_cd_matching,list_label_matching))

print("______________________________________")
print("IG")
print("Pearson Correlation", pearson_corr_ig)
print("Spearman Correlation", spearman_corr_ig)
print("Covariance", np.cov(list_ig_matching,list_label_matching))

print("______________________________________")
print(" IG + CD")
print("Pearson Correlation", pearson_corr_sum)
print("Spearman Correlation", spearman_corr_sum)
print("Covariance", np.cov(list_sum_matching,list_label_matching))

print("______________________________________")
print("Reweighted CD using IG ratio with max")
print("Pearson Correlation", pearson_corr_reweigh)
print("Spearman Correlation", spearman_corr_reweigh)
print("Covariance", np.cov(list_reweigh_matching,list_label_matching))

print("______________________________________")
print("CD + Softmax IG")
print("Pearson Correlation", pearson_corr_softmax_sum)
print("Spearman Correlation", spearman_corr_softmax_sum)
print("Covariance", np.cov(list_softmax_sum_matching,list_label_matching))

print("______________________________________")
print("Softmax IG")
print("Pearson Correlation", pearson_corr_softmax_ig)
print("Spearman Correlation", spearman_corr_softmax_ig)
print("Covariance", np.cov(list_softmax_ig_matching,list_label_matching))

print("______________________________________")
print("Softmax CD")
print("Pearson Correlation", pearson_corr_softmax_cd)
print("Spearman Correlation", spearman_corr_softmax_cd)
print("Covariance", np.cov(list_softmax_cd_matching,list_label_matching))

print("______________________________________")
print("Softmax CD + Softmax IG")
print("Pearson Correlation", pearson_corr_softmax)
print("Spearman Correlation", spearman_corr_softmax)
print("Covariance", np.cov(list_softmax_ig_cd_matching,list_label_matching))

# _________________________________________________________________________

print("--------------------------------------------------------")
print("unmatched labels")

print("______________________________________")
print("CD")
print("Pearson Correlation", pearson_corr_cd_not_matching)
print("Spearman Correlation", spearman_corr_cd_not_matching)
print("Covariance", np.cov(list_cd_not_matching,list_label_not_matching))

print("______________________________________")
print("IG")
print("Pearson Correlation", pearson_corr_ig_not_matching)
print("Spearman Correlation", spearman_corr_ig_not_matching)
print("Covariance", np.cov(list_ig_not_matching,list_label_not_matching))

print("______________________________________")
print(" IG + CD")
print("Pearson Correlation", pearson_corr_sum_not_matching)
print("Spearman Correlation", spearman_corr_sum_not_matching)
print("Covariance", np.cov(list_sum_not_matching,list_label_not_matching))

print("______________________________________")
print("Reweighted CD using IG ratio with max")
print("Pearson Correlation", pearson_corr_reweigh_not_matching)
print("Spearman Correlation", spearman_corr_reweigh_not_matching)
print("Covariance", np.cov(list_reweigh_not_matching,list_label_not_matching))

print("______________________________________")
print("CD + Softmax IG")
print("Pearson Correlation", pearson_corr_softmax_sum_not_matching)
print("Spearman Correlation", spearman_corr_softmax_sum_not_matching)
print("Covariance", np.cov(list_softmax_sum_not_matching,list_label_not_matching))

print("______________________________________")
print("Softmax IG")
print("Pearson Correlation", pearson_corr_softmax_ig_not_matching)
print("Spearman Correlation", spearman_corr_softmax_ig_not_matching)
print("Covariance", np.cov(list_softmax_ig_not_matching,list_label_not_matching))

print("______________________________________")
print("Softmax CD")
print("Pearson Correlation", pearson_corr_softmax_cd_not_matching)
print("Spearman Correlation", spearman_corr_softmax_cd_not_matching)
print("Covariance", np.cov(list_softmax_cd_not_matching,list_label_not_matching))

print("______________________________________")
print("Softmax CD + Softmax IG")
print("Pearson Correlation", pearson_corr_softmax_not_matching)
print("Spearman Correlation", spearman_corr_softmax_not_matching)
print("Covariance", np.cov(list_softmax_ig_cd_not_matching,list_label_not_matching))



