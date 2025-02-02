import os
import time
import glob
import pdb

import torch
import torch.optim as O
import torch.nn as nn

from torchtext import data
from torchtext import datasets

from model import LSTMSentiment
from sent_util import get_args, makedirs


args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inputs = data.Field(lower=args.lower)
answers = data.Field(sequential=False, unk_token=None)

train, dev, test = datasets.SST.splits(inputs, answers, fine_grained = False, train_subtrees = True,
										filter_pred=lambda ex: ex.label != 'neutral')


inputs.build_vocab(train, dev, test)
if args.word_vectors:
	if os.path.isfile(args.vector_cache):
		inputs.vocab.vectors = torch.load(args.vector_cache)
	else:
		inputs.vocab.load_vectors(args.word_vectors)
		makedirs(os.path.dirname(args.vector_cache))
		torch.save(inputs.vocab.vectors, args.vector_cache)
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
			(train, dev, test), batch_size=args.batch_size, device=device)

config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers

if args.resume_snapshot:
	model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
	model = LSTMSentiment(config)
	if args.word_vectors:
		model.embed.weight.data = inputs.vocab.vectors
		model.cuda()

criterion = nn.CrossEntropyLoss()
opt = O.Adam(model.parameters())

iterations = 0
start = time.time()
best_dev_acc = -1
train_iter.repeat = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
makedirs(args.save_path)
print(header)

all_break = False
for epoch in range(args.epochs):
	if all_break:
		break
	train_iter.init_epoch()
	n_correct, n_total = 0, 0
	for batch_idx, batch in enumerate(train_iter):
		# switch model to training mode, clear gradient accumulators
		model.train(); opt.zero_grad()

		iterations += 1

		# forward pass
		answer = model(batch)

		# calculate accuracy of predictions in the current batch
		n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
		n_total += batch.batch_size
		train_acc = 100. * n_correct/n_total

		# calculate loss of the network output with respect to training labels
		loss = criterion(answer, batch.label)

		# backpropagate and update optimizer learning rate
		loss.backward(); opt.step()

		# checkpoint model periodically
		if iterations % args.save_every == 0:
			snapshot_prefix = os.path.join(args.save_path, 'snapshot')
			snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.item(), iterations)
			torch.save(model, snapshot_path)
			for f in glob.glob(snapshot_prefix + '*'):
				if f != snapshot_path:
					os.remove(f)

		# evaluate performance on validation set periodically
		if iterations % args.dev_every == 0:

			# switch model to evaluation mode
			model.eval(); dev_iter.init_epoch()

			# calculate accuracy on validation set
			n_dev_correct, dev_loss = 0, 0
			for dev_batch_idx, dev_batch in enumerate(dev_iter):
				 answer = model(dev_batch)
				 n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
				 dev_loss = criterion(answer, dev_batch.label)
			dev_acc = 100. * n_dev_correct / len(dev)

			print(dev_log_template.format(time.time()-start,
				epoch, iterations, 1+batch_idx, len(train_iter),
				100. * (1+batch_idx) / len(train_iter), loss.item(), dev_loss.item(), train_acc, dev_acc))

			# update best valiation set accuracy
			if dev_acc > best_dev_acc:
				best_dev_acc = dev_acc
				snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
				snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}_iter_{}_model.pt'.format(dev_acc, dev_loss.item(), iterations)

				# save model, delete previous 'best_snapshot' files
				torch.save(model, snapshot_path)
				for f in glob.glob(snapshot_prefix + '*'):
					if f != snapshot_path:
						os.remove(f)

		elif iterations % args.log_every == 0:
			# print progress message
			print(log_template.format(time.time()-start,
				epoch, iterations, 1+batch_idx, len(train_iter),
				100. * (1+batch_idx) / len(train_iter), loss.item(), ' '*8, n_correct/n_total*100, ' '*12))
