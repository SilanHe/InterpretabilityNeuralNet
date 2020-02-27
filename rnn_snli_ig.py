#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import os
from argparse import ArgumentParser
import time
import glob
import torch.optim as O
import torch.nn.functional as F

from torchtext import data
from torchtext import datasets
import random
import numpy as np


import classifiers as c
import sys

# parameters

name_model = sys.argv[1]

seed=1234

# functions

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['pythonhashseed'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

seed_everything(seed)

# load arguments

parser = ArgumentParser(description='PyTorch/torchinputs SNLI example')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--d_embed', type=int, default=100)
parser.add_argument('--d_proj', type=int, default=300)
parser.add_argument('--d_hidden', type=int, default=300)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--log_every', type=int, default=50)
parser.add_argument('--lr', type=float, default=.001)
parser.add_argument('--dev_every', type=int, default=1000)
parser.add_argument('--save_every', type=int, default=1000)
parser.add_argument('--dp_ratio', type=int, default=0.2)
parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
parser.add_argument('--preserve-case', action='store_false', dest='lower')
parser.add_argument('--no-projection', action='store_false', dest='projection')
parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_path', type=str, default='results')
parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
parser.add_argument('--word_vectors', type=str, default='glove.6B.100d')
parser.add_argument('--resume_snap', type=str, default='')
args = parser.parse_args([])
print(args.batch_size,args.epochs,args.d_embed,args.d_proj,args.d_hidden,args.birnn)

# save args
torch.save(args,name_model + "_args")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inputs = data.Field(lower=True, tokenize='spacy')
answers = data.Field(sequential=False)

train, dev, test = datasets.SNLI.splits(inputs, answers)

inputs.build_vocab(train, max_size=35000, vectors="glove.6B.100d")
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=args.batch_size, device=device)


config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers
      
# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2

if args.resume_snap:
    model = torch.load(args.resume_snap, map_location=device)
else:
    embp = c.Embedder(config)
    embh = c.Embedder(config)
    model = c.SNLIClassifier(config)
    if args.word_vectors:
        embp.embedding.weight.data.copy_(inputs.vocab.vectors)
        embh.embedding.weight.data.copy_(inputs.vocab.vectors)
        embp.to(device)
        embh.to(device)
        model.to(device)

criterion = nn.CrossEntropyLoss()
opt = O.Adam(model.parameters(), lr=args.lr)

iterations = 0
start = time.time()
best_dev_acc = -1
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
makedirs(args.save_path)
print(header)

for epoch in range(args.epochs):
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iter):

        # switch model to training mode, clear gradient accumulators
        model.train(); opt.zero_grad()

        iterations += 1
        p, h = embp(batch.premise), embh(batch.hypothesis)

        # forward pass
        answer = model(p, h)

        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
        n_total += batch.batch_size
        train_acc = 100. * n_correct/n_total

        # calculate loss of the network output with respect to training labels
        loss = criterion(answer, batch.label)

        # backpropagate and update optimizer learning rate
        loss.backward(); opt.step()

        # checkpoint model periodically
        if iterations % args.save_every == 0:
            snap_prefix = os.path.join(args.save_path, 'snap')
            snap_path = snap_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.item(), iterations)
            torch.save(model, snap_path)
            for f in glob.glob(snap_prefix + '*'):
                if f != snap_path:
                    os.remove(f)

        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:

            # switch model to evaluation mode
            model.eval(); dev_iter.init_epoch()

            # calculate accuracy on validation set
            n_dev_correct, dev_loss = 0, 0
            with torch.no_grad():
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                     p, h = embp(dev_batch.premise), embh(dev_batch.hypothesis)
                     answer = model(p, h)
                     n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()) == dev_batch.label).sum().item()
                     dev_loss = criterion(answer, dev_batch.label)
            dev_acc = 100. * n_dev_correct / len(dev)

            print(dev_log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.item(), dev_loss.item(), train_acc, dev_acc))

            # update best valiation set accuracy
            if dev_acc > best_dev_acc:

                # found a model with better validation set accuracy

                best_dev_acc = dev_acc
                snap_prefix = os.path.join(args.save_path, 'best_snap')
                snap_path = snap_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc, dev_loss.item(), iterations)

                # save model, delete previous 'best_snap' files
                torch.save(model, snap_path)
                for f in glob.glob(snap_prefix + '*'):
                    if f != snap_path:
                        os.remove(f)

        elif iterations % args.log_every == 0:

            # print progress message
            print(log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.item(), ' '*8, n_correct/n_total*100, ' '*12))




# calculate accuracy on test set
n_test_correct, test_loss = 0, 0
with torch.no_grad():
    for test_batch_idx, test_batch in enumerate(test_iter):
         p, h = embp(test_batch.premise), embh(test_batch.hypothesis)
         answer = model(p, h)
         n_test_correct += (torch.max(answer, 1)[1].view(test_batch.label.size()) == test_batch.label).sum().item()
         #test_loss = criterion(answer, test_batch.label)
test_acc = 100. * n_test_correct / len(test)

print('Test accuracy : %f'%(test_acc))

# save the model for future querying

torch.save(embp,name_model + "_embp")
torch.save(embp,name_model + "_embh")
torch.save(model,name_model)





