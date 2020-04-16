import torch
import torch.nn as nn
from torch import Tensor
from torchtext.data.batch import Batch
from torch.autograd import Variable
import pdb

class LSTMSentiment(nn.Module):

	def __init__(self, config):
		super(LSTMSentiment, self).__init__()
		self.hidden_dim = config.d_hidden
		self.vocab_size = config.n_embed
		self.emb_dim = config.d_embed
		self.num_out = config.d_out
		self.batch_size = config.batch_size
		self.use_gpu = True #config.use_gpu
		self.num_labels = 2
		self.embed = nn.Embedding(self.vocab_size, self.emb_dim)
		self.lstm = nn.LSTM(input_size = self.emb_dim, hidden_size = self.hidden_dim)
		self.hidden_to_label = nn.Linear(self.hidden_dim, self.num_labels)

	def forward(self, batch):
		

		# check if a batch or an input tensor
		if isinstance(batch,Tensor):
			vecs = batch
			if self.use_gpu:
				self.hidden = (Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
								Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))
			else:
				self.hidden = (Variable(torch.zeros(1, 1, self.hidden_dim)),
								Variable(torch.zeros(1, 1, self.hidden_dim)))
		else: # assume batch has text field and is a tensor
			vecs = self.embed(batch.text)
			if self.use_gpu:
				self.hidden = (Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim).cuda()),
								Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim).cuda()))
			else:
				self.hidden = (Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim)),
								Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim)))

		lstm_out, self.hidden = self.lstm(vecs, self.hidden)
		logits = self.hidden_to_label(lstm_out[-1])
		#log_probs = self.log_softmax(logits)
		#return log_probs
		return logits

