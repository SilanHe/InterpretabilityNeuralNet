import os
import pdb
import torch
import numpy as np
from argparse import ArgumentParser
from torchtext import data, datasets
from torchtext.data.batch import Batch
from torch import Tensor
from scipy.special import expit as sigmoid
import random
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import nltk
import nltk.corpus


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get inputs, answers, training set iterator and dev set iterator
def get_sst():    
	inputs = data.Field(lower='preserve-case')
	answers = data.Field(sequential=False, unk_token=None)

	# build with subtrees so inputs are right
	train_s, dev_s, test_s = datasets.SST.splits(inputs, answers, fine_grained = False, train_subtrees = True,
										   filter_pred=lambda ex: ex.label != 'neutral')
	inputs.build_vocab(train_s, dev_s, test_s)
	answers.build_vocab(train_s)
	
	# rebuild without subtrees to get longer sentences
	train, dev, test = datasets.SST.splits(inputs, answers, fine_grained = False, train_subtrees = False,
									   filter_pred=lambda ex: ex.label != 'neutral')
	
	train_iter, dev_iter, test_iter = data.BucketIterator.splits(
			(train, dev, test), batch_size=1, device=device)

	return inputs, answers, train_iter, dev_iter

# get model
def get_model(snapshot_file):
	print('loading', snapshot_file)
	try:  # load onto gpu
		model = torch.load(snapshot_file)
		print('loaded onto gpu...')
	except:  # load onto cpu
		model = torch.load(snapshot_file, map_location=lambda storage, loc: storage)
		print('loaded onto cpu...')
	return model

# gets the batches of the specified dset, by default 'train'
# batch_nums is a list of int, each of which represent an index you wish to retrieve
# train_iterator is our iterator from get_sst()
# dev_iterator is our iterator from get_sst()
def get_batches(batch_nums, train_iterator, dev_iterator, dset='train'):
	print('getting batches...')
	np.random.seed(13)
	random.seed(13)
	
	# pick data_iterator
	if dset=='train':
		data_iterator = train_iterator
	elif dset=='dev':
		data_iterator = dev_iterator
	
	# actually get batches
	num = 0
	batches = {}
	data_iterator.init_epoch() 
	for batch_idx, batch in enumerate(data_iterator):
		if batch_idx == batch_nums[num]:
			batches[batch_idx] = batch
			num +=1 

		if num == max(batch_nums):
			break
		elif num == len(batch_nums):
			print('found them all')
			break
	return batches

# gets the batches from data_iterator, overloaded version of above function
def get_batches_iterator(batch_nums, data_iterator):
	print('getting batches...')
	np.random.seed(13)
	random.seed(13)
	
	# actually get batches that match indices in batch_nums
	num = 0
	batches = {}
	data_iterator.init_epoch() 
	for batch_idx, batch in enumerate(data_iterator):
		if batch_idx == batch_nums[num]:
			batches[batch_idx] = batch
			num +=1 

		if num == max(batch_nums):
			break
		elif num == len(batch_nums):
			print('found them all')
			break
	return batches

# batch of [start, stop) with unigrams working
# batch here refers to input instance, which is a bunch of strings as a list of string batch.text
# batch is a LongTensor
# model: our sst model
def CD(batch, model, start, stop):
	weights = model.lstm.state_dict()

	# Index one = word vector (i) or hidden state (h), index two = gate
	W_ii, W_if, W_ig, W_io = np.split(weights['weight_ih_l0'].cpu(), 4, 0)
	W_hi, W_hf, W_hg, W_ho = np.split(weights['weight_hh_l0'].cpu(), 4, 0)
	b_i, b_f, b_g, b_o = np.split(weights['bias_ih_l0'].cpu().numpy() + weights['bias_hh_l0'].cpu().numpy(), 4)
	
	if isinstance(batch, data.Batch):
		word_vecs = model.embed(batch.text)[:,0].data
	else:
		word_vecs = model.embed(batch)[:,0].data

	T = word_vecs.size(0)
	word_vecs = [word_vec.cpu() for word_vec in word_vecs]
	relevant = np.zeros((T, model.hidden_dim))
	irrelevant = np.zeros((T, model.hidden_dim))
	relevant_h = np.zeros((T, model.hidden_dim))
	irrelevant_h = np.zeros((T, model.hidden_dim))

	for i in range(T):
		if i > 0:
			prev_rel_h = relevant_h[i - 1]
			prev_irrel_h = irrelevant_h[i - 1]
		else:
			prev_rel_h = np.zeros(model.hidden_dim)
			prev_irrel_h = np.zeros(model.hidden_dim)

		rel_i = np.dot(W_hi, prev_rel_h)
		rel_g = np.dot(W_hg, prev_rel_h)
		rel_f = np.dot(W_hf, prev_rel_h)
		rel_o = np.dot(W_ho, prev_rel_h)
		irrel_i = np.dot(W_hi, prev_irrel_h)
		irrel_g = np.dot(W_hg, prev_irrel_h)
		irrel_f = np.dot(W_hf, prev_irrel_h)
		irrel_o = np.dot(W_ho, prev_irrel_h)

		if i >= start and i <= stop:
			rel_i = rel_i + np.dot(W_ii, word_vecs[i])
			rel_g = rel_g + np.dot(W_ig, word_vecs[i])
			rel_f = rel_f + np.dot(W_if, word_vecs[i])
			rel_o = rel_o + np.dot(W_io, word_vecs[i])
		else:
			irrel_i = irrel_i + np.dot(W_ii, word_vecs[i])
			irrel_g = irrel_g + np.dot(W_ig, word_vecs[i])
			irrel_f = irrel_f + np.dot(W_if, word_vecs[i])
			irrel_o = irrel_o + np.dot(W_io, word_vecs[i])

		rel_contrib_i, irrel_contrib_i, bias_contrib_i = decomp_three(rel_i, irrel_i, b_i, sigmoid)
		rel_contrib_g, irrel_contrib_g, bias_contrib_g = decomp_three(rel_g, irrel_g, b_g, np.tanh)

		relevant[i] = rel_contrib_i * (rel_contrib_g + bias_contrib_g) + bias_contrib_i * rel_contrib_g
		irrelevant[i] = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + (rel_contrib_i + bias_contrib_i) * irrel_contrib_g

		if i >= start and i < stop:
			relevant[i] += bias_contrib_i * bias_contrib_g
		else:
			irrelevant[i] += bias_contrib_i * bias_contrib_g

		if i > 0:
			rel_contrib_f, irrel_contrib_f, bias_contrib_f = decomp_three(rel_f, irrel_f, b_f, sigmoid)
			relevant[i] += (rel_contrib_f + bias_contrib_f) * relevant[i - 1]
			irrelevant[i] += (rel_contrib_f + irrel_contrib_f + bias_contrib_f) * irrelevant[i - 1] + irrel_contrib_f * relevant[i - 1]

		o = sigmoid(np.dot(W_io, word_vecs[i]) + np.dot(W_ho, prev_rel_h + prev_irrel_h) + b_o)
		rel_contrib_o, irrel_contrib_o, bias_contrib_o = decomp_three(rel_o, irrel_o, b_o, sigmoid)
		new_rel_h, new_irrel_h = decomp_tanh_two(relevant[i], irrelevant[i])
		#relevant_h[i] = new_rel_h * (rel_contrib_o + bias_contrib_o)
		#irrelevant_h[i] = new_rel_h * (irrel_contrib_o) + new_irrel_h * (rel_contrib_o + irrel_contrib_o + bias_contrib_o)
		relevant_h[i] = o * new_rel_h
		irrelevant_h[i] = o * new_irrel_h

	W_out = model.hidden_to_label.weight.data.cpu()
	
	# Sanity check: scores + irrel_scores should equal the LSTM's output minus model.hidden_to_label.bias
	scores = np.dot(W_out, relevant_h[T - 1])
	irrel_scores = np.dot(W_out, irrelevant_h[T - 1])

	return scores, irrel_scores


	
# batch: batch
# model: our sst model
# inputs: vocab for encoding input sentence
# answers: vocab for encoding labels
def CD_unigram(batch, model, inputs, answers, output = True):

	# local function for formating score for panda printing
	def format_score(score):
		return score[0] - score[1]

	# set up
	if isinstance(batch,Batch):
		len_batch = len(batch.text)
		text = batch.text.data[:, 0]
		words = [inputs.vocab.itos[i] for i in text]

		with torch.no_grad():
			model.eval()
			pred=torch.argmax(model(batch))
		model.train()

	elif isinstance(batch,Tensor):
		text = batch.data[:, 0]
		len_batch = len(text)
		words = [inputs.vocab.itos[i] for i in text]
		x = model.embed(batch)

		with torch.no_grad():
			model.eval()
			pred=torch.argmax(model(x))
		model.train()

	scores = list()
	scores_irrel = list()
	

	# print sentence + CD for whole sentence
	sentence, sentence_irrel = CD(batch, model, start = 0, stop = len_batch)
	print(' '.join(words[:len_batch]), sentence[0] - sentence[1])

	# for each word in the batch, get our scores by calling CD on a single word
	for i in range(len_batch):
		score, score_irrel = CD(batch, model, i, i)
		scores.append(score)
		scores_irrel.append(score_irrel)

	# print using panda
	formatted_score = [format_score(s) for s in scores]
	if output:
		df = pd.DataFrame(index=['SST','ContextualDecomp'], columns=list(range(len_batch)), data=[words, formatted_score])

		with pd.option_context('display.max_rows', None, 'display.max_columns', 30):
			print(df)
		if isinstance(batch,Batch):
			print("TRUE Label : ",answers.vocab.itos[batch.label.data[0]])
		print("PREDICTED Label : ", answers.vocab.itos[pred.item()])

		# visual delimiter so its easier to see different examples
		print("_____________________________")

	return answers.vocab.itos[pred.item()], formatted_score
	
def decomp_three(a, b, c, activation):
	a_contrib = 0.5 * (activation(a + c) - activation(c) + activation(a + b + c) - activation(b + c))
	b_contrib = 0.5 * (activation(b + c) - activation(c) + activation(a + b + c) - activation(a + c))
	return a_contrib, b_contrib, activation(c)

def decomp_tanh_two(a, b):
	return 0.5 * (np.tanh(a) + (np.tanh(a + b) - np.tanh(b))), 0.5 * (np.tanh(b) + (np.tanh(a + b) - np.tanh(a)))
	

def makedirs(name):
	"""helper function for python 2 and 3 to call os.makedirs()
	   avoiding an error if the directory to be created already exists"""

	import os, errno

	try:
		os.makedirs(name)
	except OSError as ex:
		if ex.errno == errno.EEXIST and os.path.isdir(name):
			# ignore existing directory
			pass
		else:
			# a different error happened
			raise

# batch: string inside batch object
# model: our sst model
# inputs: vocab for encoding input sentence
# answers: vocab for encoding labels
def integrated_gradients_unigram(batch, model, inputs, answers, output = True):

	# set up
	if isinstance(batch,Batch):
		text = batch.text.data[:, 0]
		words = [inputs.vocab.itos[i] for i in text]
		x = model.embed(batch.text)
		len_batch = len(batch.text)

	elif isinstance(batch,Tensor):
		text = batch.data[:, 0]
		words = [inputs.vocab.itos[i] for i in text]
		x = model.embed(batch)
		len_batch = len(words)
	
	T = x.size(0)
	word_vecs = [word_vec.cpu() for word_vec in x]

	x_dash = torch.zeros_like(x)
	sum_grad = None
	grad_array = None
	x_array = None

	# get Predicted label
	with torch.no_grad():
		model.eval()
		pred=torch.argmax(model(x))
	model.train()

	# ig
	for k in range(T):
		model.zero_grad()
		step_input = x_dash + k * (x - x_dash) / T
		step_output = model(step_input)
		step_pred = torch.argmax(step_output)
		step_grad = torch.autograd.grad(step_output[0][pred.item()], x)[0]
		if sum_grad is None:
			sum_grad = step_grad
			grad_array = step_grad
			x_array = step_input
		else:
			sum_grad += step_grad
			grad_array = torch.cat([grad_array, step_grad])
			x_array = torch.cat([x_array, step_input])

	sum_grad = sum_grad / T
	sum_grad = sum_grad * (x - x_dash)
	sum_grad = sum_grad.sum(dim=2)

	relevances = sum_grad.detach().cpu().numpy()


	try:
		relevances = list(np.round(np.reshape(relevances,len(words)),3))
		if output:
			df = pd.DataFrame(index=['Sentence','IntegGrad'], columns=list(range(len(words))), data=[words, relevances])
			print("Sentence : %s"%(' '.join(words)))
			with pd.option_context('display.max_rows', None, 'display.max_columns', 30):
				print(df)
			
			if isinstance(batch,Batch):
				print("TRUE Label : %s"%(answers.vocab.itos[batch.label.data[0]]))
			print("PREDICTED Label : %s"%(answers.vocab.itos[pred.item()]))
			# visual delimiter so its easier to see different examples
			print("_____________________________")
		return answers.vocab.itos[pred], relevances
	except:
		if output:
			print("*****Error*******")
		return answers.vocab.itos[pred], []

# batch: string inside batch object
# model: our sst model
# inputs: vocab for encoding input sentence
# answers: vocab for encoding labels
def integrated_gradients_sum_baseline(batch, model, inputs, answers, start, stop, output = True):

	def print_IG(text,score,label):

		# print using panda
		print(' '.join(text))
		print("IG score", join)
		print(label)

		# visual delimiter so its easier to see different examples
		print("-------------------")

	# set up
	if isinstance(batch,Batch):
		text = batch.text.data[:, 0]
		words = [inputs.vocab.itos[i] for i in text]
		x = model.embed(batch.text)
		len_batch = len(batch.text)

	elif isinstance(batch,Tensor):
		text = batch.data[:, 0]
		words = [inputs.vocab.itos[i] for i in text]
		x = model.embed(batch)
		len_batch = len(words)

	label, relevances = integrated_gradients_unigram(batch, model, inputs, answers, output = False)

	score_ig = sum(relevances[start:stop+1])
	if output:
		print_IG(words[start:stop+1],score_ig,label)

	return label, score_ig

# batch: string inside batch object
# model: our sst model
# inputs: vocab for encoding input sentence
# answers: vocab for encoding labels
def integrated_gradients_function(batch, model, inputs, answers, start, stop, fun, output = True):

	def print_IG(text,score,label):

		# print using panda
		print(' '.join(text))
		print("IG score", join)
		print(label)

		# visual delimiter so its easier to see different examples
		print("-------------------")

	# set up
	if isinstance(batch,Batch):
		text = batch.text.data[:, 0]
		words = [inputs.vocab.itos[i] for i in text]
		x = model.embed(batch.text)
		len_batch = len(batch.text)

	elif isinstance(batch,Tensor):
		text = batch.data[:, 0]
		words = [inputs.vocab.itos[i] for i in text]
		x = model.embed(batch)
		len_batch = len(words)

	label, relevances = integrated_gradients_unigram(batch, model, inputs, answers, output = False)

	score_ig = fun(relevances[start:stop+1])
	if output:
		print_IG(words[start:stop+1],score_ig,label)

	return label, score_ig


# returns predictions
def eval_model(batch, model, answers):

	if isinstance(batch,Tensor):
		x = model.embed(batch)
	else:
		x = model.embed(batch.text)

	# get Predicted label
	with torch.no_grad():
		model.eval()
		pred=torch.argmax(model(x))
	model.train()

	return answers.vocab.itos[pred], pred

# returns true if the true and predicted labels are different
def diff_predicted_label(batch, model, answers):

	true_label = answers.vocab.itos[batch.label.data[0]]
	predicted_label, _ = eval_model(batch, model, answers)

	return true_label != predicted_label

def get_sst_PTB(path = "/Users/silanhe/Documents/McGill/Grad/WINTER2020/NLU/ig/data/trees"):
	sst_reader = nltk.corpus.BracketParseCorpusReader(path, ".*.txt")
	sst_sentences = sst_reader.sents("test.txt")
	sst = sst_reader.parsed_sents("test.txt")

	return sst_sentences, sst

# in this function, 
# batch is a list of str
# model is the network
# inputs is the vocab
# node is the root of the tree in ptb format
# returns list of scores and labels
def travelTreeUnigram(batch,model,inputs,answers,node, output = True):

	def convert_PTB_label(label):
		if label <= 2:
			return "negative"
		else:
			return "positive"

	# convert batch to tensor
	vector = [[inputs.vocab.stoi[word]] for word in batch]
	word_tensor = torch.LongTensor(vector).to(device)

	# set up
	len_batch = len(batch)
	index_words = 0

	# get list of scores + labels
	list_scores_ig = list()
	list_scores_cd = list()
	list_labels = list()
	
	def dfs(node,score):
		nonlocal word_tensor,model,index_words,list_scores_ig,list_scores_cd,list_labels
		if isinstance(node,str):
			list_labels.append(score)
			index_words += 1
		else:
			label = int(node.label())
			if len(node) > 0:
				dfs(node[0],label)
			if len(node) > 1:
				dfs(node[1],label)

	def print_labels(sentence,list_scores):
		df = pd.DataFrame(index=['SST','Labels'], columns=list(range(len_batch)), data=[sentence, list_scores])

		with pd.option_context('display.max_rows', None, 'display.max_columns', 30):
			print(df)

		print("_____________________________")

	# get predicted label
	predicted_label, _ = eval_model(word_tensor,model,answers)
	if not isinstance(node,str): # if not str, this shouldnt happen
		ground_truth_label = convert_PTB_label(int(node.label()))
	
	if node:
		dfs(node, 0)
	else:
		print("ERROR")

	_, list_ig = integrated_gradients_unigram(word_tensor, model, inputs, answers)
	_, list_cd = CD_unigram(word_tensor, model, inputs, answers)
	list_scores_ig += list_ig
	list_scores_cd += list_cd

	if output:
		print("______________________________________")
		print_labels(batch,list_labels)
		
		print("______________________________________")

	return list_scores_ig, list_scores_cd, list_labels, predicted_label == ground_truth_label



# in this function, 
# batch is a list of str
# model is the network
# inputs is the vocab
# node is the root of the tree in ptb format
# returns list of scores and labels
def travelTree(batch,model,inputs,node,output = True):

	# local function for formating score for panda printing
	def format_score(score):
		return score[0] - score[1]

	def print_CD(text,score,label):

		# print using panda
		formatted_score = format_score(score)
		print(' '.join(text))
		print("CD score", formatted_score)
		print("positive" if label >2 else "negative")

		# visual delimiter so its easier to see different examples
		print("-------------------")
	
	index_words = 0

	# convert batch to tensor
	vector = [[inputs.vocab.stoi[word]] for word in batch]
	word_tensor = torch.LongTensor(vector).to(device)

	# set up
	len_batch = len(batch)

	# get list of scores + labels
	list_scores = list()
	list_labels = list()
	
	def dfs(node):
		nonlocal word_tensor,model,index_words,list_scores,list_labels
		if isinstance(node,str):
			list_return = [index_words]
			index_words += 1
			return list_return

		else:
			label = int(node.label())
			len_node = len(node)
			
			subtree_list_words = []
			if len(node) > 0:
				subtree_list_words += dfs(node[0])
			if len(node) > 1:
				subtree_list_words += dfs(node[1])
			
			# get CD score
			start = min(subtree_list_words)
			end = max(subtree_list_words)
			score, _ = CD(word_tensor, model, start, end)
			if output:
				print_CD(batch[start:end + 1], score, label)
			list_scores.append(format_score(score))
			list_labels.append(label)

			return subtree_list_words

	if output:
		print("______________________________________")
	if node:
		dfs(node)
	else:
		print("ERROR")
	
	if output:
		print("______________________________________")

	return list_scores, list_labels

# in this function, 
# batch is a list of str
# model is the network
# inputs is the vocab
# node is the root of the tree in ptb format
# returns list of scores and labels
def travelTree_IG_CD(batch,model,inputs, answers, node, fun, output = True):

	# local function for formating score for panda printing
	def format_score(score):
		return score[0] - score[1]

	def print_scores( text, score_cd, score_ig, label):

		# print using panda
		formatted_score = format_score(score_cd)
		print(' '.join(text))
		print("CD score", formatted_score)
		print("IG score", score_ig)
		print("positive" if label >2 else "negative")

		# visual delimiter so its easier to see different examples
		print("-------------------")

	def convert_PTB_label(label):
		if label <= 2:
			return "negative"
		else:
			return "positive"
	
	index_words = 0

	# convert batch to tensor
	vector = [[inputs.vocab.stoi[word]] for word in batch]
	word_tensor = torch.LongTensor(vector).to(device)

	# set up
	len_batch = len(batch)

	# get list of scores + labels
	list_scores_cd = list()
	list_scores_ig = list()
	list_labels = list()
	
	def dfs(node):
		nonlocal word_tensor,model,index_words,list_scores_cd, list_scores_ig,list_labels
		if isinstance(node,str):
			list_return = [index_words]
			index_words += 1
			return list_return

		else:
			label = int(node.label())
			len_node = len(node)
			
			subtree_list_words = []
			if len(node) > 0:
				subtree_list_words += dfs(node[0])
			if len(node) > 1:
				subtree_list_words += dfs(node[1])
			
			# get CD score
			start = min(subtree_list_words)
			end = max(subtree_list_words)
			cd_score, _ = CD(word_tensor, model, start, end)

			# get IG score
			_, ig_score = integrated_gradients_function(word_tensor, model, inputs, answers, start, end, fun, output = False)
			
			list_scores_cd.append(format_score(cd_score))
			list_scores_ig.append(ig_score)
			list_labels.append(label)

			if output:
				print_scores(batch[start:end + 1], cd_score, ig_score, label)

			return subtree_list_words

	# get predicted label
	predicted_label, _ = eval_model(word_tensor,model,answers)
	if not isinstance(node,str): # if not str, this shouldnt happen
		ground_truth_label = convert_PTB_label(int(node.label()))

	if output:
		print("______________________________________")
	if node:
		dfs(node)
	else:
		print("ERROR")
	
	if output:
		print("______________________________________")

	return list_scores_cd, list_scores_ig, list_labels, predicted_label == ground_truth_label

def get_args():
	parser = ArgumentParser(description='PyTorch/torchtext SST')
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--batch_size', type=int, default=50)
	parser.add_argument('--d_embed', type=int, default=300)
	parser.add_argument('--d_proj', type=int, default=300)
	parser.add_argument('--d_hidden', type=int, default=128)
	parser.add_argument('--n_layers', type=int, default=1)
	parser.add_argument('--log_every', type=int, default=1000)
	parser.add_argument('--lr', type=float, default=.001)
	parser.add_argument('--dev_every', type=int, default=1000)
	parser.add_argument('--save_every', type=int, default=1000)
	parser.add_argument('--dp_ratio', type=int, default=0.2)
	parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
	parser.add_argument('--preserve-case', action='store_false', dest='lower')
	parser.add_argument('--no-projection', action='store_false', dest='projection')
	parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--save_path', type=str, default='results_sst')
	parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
	parser.add_argument('--word_vectors', type=str, default='glove.6B.300d')
	parser.add_argument('--resume_snapshot', type=str, default='')
	args = parser.parse_args()
	return args
