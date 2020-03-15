import os
import pdb
import torch
import numpy as np
from argparse import ArgumentParser
from torchtext import data, datasets
from scipy.special import expit as sigmoid
import random
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# model: our sst model
def CD(batch, model, start, stop):
    weights = model.lstm.state_dict()

    # Index one = word vector (i) or hidden state (h), index two = gate
    W_ii, W_if, W_ig, W_io = np.split(weights['weight_ih_l0'].cpu(), 4, 0)
    W_hi, W_hf, W_hg, W_ho = np.split(weights['weight_hh_l0'].cpu(), 4, 0)
    b_i, b_f, b_g, b_o = np.split(weights['bias_ih_l0'].cpu().numpy() + weights['bias_hh_l0'].cpu().numpy(), 4)
    word_vecs = model.embed(batch.text)[:,0].data
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


    
# batch: string inside batch object
# model: our sst model
# inputs: vocab for encoding input sentence
# answers: vocab for encoding labels
def CD_unigram(batch, model, inputs, answers):

    # local function for formating score for panda printing
    def format_score(score):
        return score[0] - score[1]

    # set up
    len_batch = len(batch.text)
    text = batch.text.data[:, 0]
    words = [inputs.vocab.itos[i] for i in text]
    scores = list()
    scores_irrel = list()

    # get predicted label
    x = model.embed(batch.text)[:,0].data
    T = x.size(0)
    word_vecs = [word_vec.cpu() for word_vec in x]

    with torch.no_grad():
        model.eval()
        pred=torch.argmax(model(x))
    model.train()

    # print sentence + CD for whole sentence
    sentence, sentence_irrel = sent_util.CD(batch, model, start = 0, stop = len_batch)
    print(' '.join(words[:len_batch]), sentence[0] - sentence[1])

    # for each word in the batch, get our scores by calling CD on a single word
    for i in range(len_batch):
        score, score_irrel = sent_util.CD(batch, model, i, i)
        scores.append(score)
        scores_irrel.append(score_irrel)

    # print using panda
    formatted_score = [format_score(s) for s in scores]
    df = pd.DataFrame(index=['SST','ContextualDecomp'], columns=list(range(len_batch)), data=[words, formatted_score])

    with pd.option_context('display.max_rows', None, 'display.max_columns', 30):
        print(df)

    print("PREDICTED Label : ", answers.vocab.itos[pred])
    print("TRUE Label : ",answers.vocab.itos[batch.label.data[0]])

    # visual delimiter so its easier to see different examples
    print("_____________________________")
    
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
def integrated_gradients_unigram(batch, model, inputs, answers):
    
    # set up
    len_batch = len(batch.text)
    text = batch.text.data[:, 0]
    words = [inputs.vocab.itos[i] for i in text]

    x = model.embed(batch.text)[:,0].data
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
        step_grad = torch.autograd.grad(step_output[pred], x)[0]
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
        df = pd.DataFrame(index=['Sentence','IntegGrad'], columns=list(range(len(words))), data=[words, relevances])
        print("Sentence : %s"%(s))
        with pd.option_context('display.max_rows', None, 'display.max_columns', 30):
            print(df)
        print("PREDICTED Label : %s"%(answers.vocab.itos[pred]))
        print("TRUE Label : %s"%(answers.vocab.itos[batch.label.data[0]]))
        return answers.vocab.itos[pred], relevances
    except:
        print("*****Error*******")
        return answers.vocab.itos[pred], []

    # visual delimiter so its easier to see different examples
    print("_____________________________")

def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SST')
    parser.add_argument('--epochs', type=int, default=5)
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
