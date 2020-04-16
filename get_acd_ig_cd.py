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
import acd
from visualization import viz_1d as viz

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Batch:
	text = torch.zeros(1).to(device)

# function
def magnitude(l):
	xs = np.array(l)
	xs_abs = np.abs(xs)
	max_index = np.argmax(xs_abs)
	x = xs[max_index]

	return x

# base parameters
sweep_dim = 1 # how large chunks of text should be considered (1 for words)
method = 'cd' # build_up, break_down, cd
percentile_include = 99.5 # keep this very high so we don't add too many words at once
num_iters = 25 # maximum number of iterations (rarely reached)

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

	sentence = [word.lower() for word in sst_sentences[index]]
	vector = [[inputs.vocab.stoi[word]] for word in sentence]
	word_tensor = torch.LongTensor(vector).to(device)
	batch = Batch()
	batch.text = word_tensor

	label, label_pred = sent_util.eval_model(batch, model, answers)

	# cd,ig,label,matching = sent_util.travelTree_IG_CD(sentence, model, inputs, answers, tree, sum) # get ig phrase level scores as well using sum baseline

	# agglomerate
	lists = acd.agg_1d.agglomerate(model, batch, percentile_include, method, sweep_dim, # only works for sweep_dim = 1
	                    label_pred, num_iters=num_iters, device=device) # see agg_1d.agglomerate to understand what this dictionary contains
	lists = acd.agg_1d.collapse_tree(lists) # don't show redundant joins

	# visualize
	print("CD")
	viz.word_heatmap(sentence, lists, label_pred, label, fontsize=9)

	# agglomerate
	lists = acd.agg_1d.agglomerate(model, batch, percentile_include, 'occlusion', sweep_dim, # only works for sweep_dim = 1
	                    label_pred, num_iters=num_iters, device=device) # see agg_1d.agglomerate to understand what this dictionary contains
	lists = acd.agg_1d.collapse_tree(lists) # don't show redundant joins

	print("IG")
	viz.word_heatmap(sentence, lists, label_pred, label, fontsize=9)
