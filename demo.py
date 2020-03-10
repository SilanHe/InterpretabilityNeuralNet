#!/usr/bin/env python
# coding: utf-8


import numpy as np
import sys
from os.path import join as oj
sys.path.insert(1, oj(sys.path[0], 'train_model'))

from train_model import sent_util

import torch
from torchtext import data, datasets



# To train model, first run 'train.py' from train_model dir

# get model
snapshot_dir = 'results_sst/'

snapshot_file = oj(snapshot_dir, 
					'best_snapshot_devacc_79.35779571533203_devloss_0.41613781452178955_iter_9000_model.pt')
model = sent_util.get_model(snapshot_file)

# get data
inputs, answers, train_iterator, dev_iterator = sent_util.get_sst()



# Find sentence used in figure 2
batch_nums = list(range(6920))
data = sent_util.get_batches(batch_nums, train_iterator, dev_iterator) 
for ind in range(6919):
	text = data[ind].text.data[:, 0]
	words = [inputs.vocab.itos[i] for i in text]
	if words[0] == 'it' and words[1] == "'s" and words[2] == 'easy':
		high_level_comp_ind = ind
		break


# Produce CD importance scores for phrases used in figure 2
pos, pos_irrel = sent_util.CD(data[high_level_comp_ind], model, start = 0, stop = 15)
print(' '.join(words[:16]), pos[0] - pos[1])
neg, neg_irrel = sent_util.CD(data[high_level_comp_ind], model, start = 16, stop = 26)
print(' '.join(words[16:]), neg[0] - neg[1])


# it 's easy to love robin tunney -- she 's pretty and she can act -- 0.545045315382
# but it gets harder and harder to understand her choices . -1.22609390466


# Sanity check: CD is a decomposition, so an effective way to check for bugs is to verify that the decomposition holds (up to numerical errors)
print(pos + pos_irrel)
linear_bias = model.hidden_to_label.bias.data.cpu().numpy()
print((model(data[high_level_comp_ind]).data.cpu().numpy() - linear_bias)[0])

for ind in range(5):
	CD_unigram(data[ind], model)

def CD_unigram(batch, model):
	len_batch = len(batch)
	text = batch.text.data[:, 0]
	words = [inputs.vocab.itos[i[0]] for i in text]
	scores = list()
	scores_irrel = list()

	for i in range(len_batch):
		score, score_irrel = sent_util.CD(batch, model, i, i)
		scores.append(score)
		scores_irrel.append(score_irrel)

	print(' '.join(words[:16]), scores)
	print("_____________________________")

