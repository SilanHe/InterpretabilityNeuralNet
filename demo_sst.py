#!/usr/bin/env python
# coding: utf-8
import numpy as np
import sys
from os.path import join
sys.path.insert(1, join(sys.path[0], 'train_model'))

from train_model import sent_util

import torch
from torchtext import data, datasets
import pandas as pd

# To train model, first run 'train.py' from train_model dir

# get model path, OS safe
snapshot_dir = 'models/'
snapshot_file = join(snapshot_dir, 'best_rnn_model_sst.tar')

# get model
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

# our unigram stuff
sent_util.CD_unigram(data[high_level_comp_ind], model, inputs, answers)
sent_util.integrated_gradients_unigram(data[high_level_comp_ind], model, inputs, answers)

for ind in range(0,20):
	sent_util.CD_unigram(data[ind], model, inputs, answers)

for ind in range(0,20):
	sent_util.integrated_gradients_unigram(data[ind], model, inputs, answers)





