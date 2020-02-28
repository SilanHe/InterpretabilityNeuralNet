import spacy
import json

import torch
import torch.nn as nn
import os
from argparse import ArgumentParser

from torchtext import data
from torchtext import datasets
import numpy as np

import classifiers as c
import sys

name_model = sys.argv[1]
model = torch.load(name_model)
args = torch.load(name_model + "_args")
embp = torch.load(name_model + "_embp")
embh = torch.load(name_model + "_embh")

nlp = spacy.load('en')
#model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inputs = data.Field(lower=True, tokenize='spacy')
answers = data.Field(sequential=False)

train, dev, test = datasets.SNLI.splits(inputs, answers)

inputs.build_vocab(train, max_size=35000, vectors="glove.6B.100d")
answers.build_vocab(train)

def embed_pair(s1_premise, s2_hypothesis, label):
    tmap={}
    tmap['sentence1'],tmap['sentence2'],tmap['gold_label'] = s1_premise,s2_hypothesis,label
    with open('./.data/snli/snli_1.0/result.jsonl', 'w') as fp:
        json.dump(tmap, fp)
    a,b,c = datasets.SNLI.splits(inputs, answers, train='result.jsonl', validation='result.jsonl', test='result.jsonl')
    a_iter,b_iter,c_iter = data.BucketIterator.splits((a,b,c), batch_size=128, device=device)
    batches=[(idx, batch) for idx, batch in enumerate(c_iter)]
    embp.eval()
    embh.eval()
    p_emb, h_emb = embp(batches[0][1].premise), embp(batches[0][1].hypothesis)
    return p_emb, h_emb

def predict_entailment(s1_premise,s2_hypothesis,label=''):
    p_emb, h_emb = embed_pair(s1_premise, s2_hypothesis,label)
    with torch.no_grad():
        model.eval()
        answer = model(p_emb, h_emb)
    return answers.vocab.itos[torch.max(answer, 1)[1].item()]


# test queries
print("A black race car starts up in front of a crowd of people.","A man is driving down a lonely road.",predict_entailment("A black race car starts up in front of a crowd of people.","A man is driving down a lonely road."))
print("A soccer game with multiple males playing.","Some men are playing a sport.",predict_entailment("A soccer game with multiple males playing.","Some men are playing a sport."))
print("A smiling costumed woman is holding an umbrella.","A happy woman in a fairy costume holds an umbrella.",predict_entailment("A smiling costumed woman is holding an umbrella.","A happy woman in a fairy costume holds an umbrella."))
print("A person on a horse jumps over a broken down airplane.","A person is training his horse for a competition.",predict_entailment("A person on a horse jumps over a broken down airplane.","A person is training his horse for a competition."))
print("A person on a horse jumps over a broken down airplane.","A person is at a diner, ordering an omelette.",predict_entailment("A person on a horse jumps over a broken down airplane.","A person is at a diner, ordering an omelette."))
print("A person on a horse jumps over a broken down airplane.","A person is outdoors, on a horse.",predict_entailment("A person on a horse jumps over a broken down airplane.","A person is outdoors, on a horse."))
print("A person on a horse jumps over a broken down airplane.","A person is indoors, on a horse.",predict_entailment("A person on a horse jumps over a broken down airplane.","A person is indoors, on a horse."))
print("A person on a horse jumps over a broken down airplane.","A person is outside, on a horse.",predict_entailment("A person on a horse jumps over a broken down airplane.","A person is outside, on a horse."))
print("A person on a horse jumps over a sofa.","A person is outside, on a horse.",predict_entailment("A person on a horse jumps over a sofa.","A person is outside, on a horse."))
print("A person is beside a horse.","A person is outside, on a horse.",predict_entailment("A person is beside a horse.","A person is outside, on a horse."))
print("A person is beside a boy.","A person is outside, on a horse.",predict_entailment("A person is beside a boy.","A person is outside, on a horse."))


import pandas as pd
def integrated_gradients(s1_premise, s2_hypothesis, m=300):
    p, h = embed_pair(s1_premise,s2_hypothesis,'')
    p_dash, h_dash = torch.zeros_like(p), torch.zeros_like(h)
    sum_grad = None
    with torch.no_grad():
        model.eval()
        pred=torch.argmax(model(p, h))
    model.train()
    for k in range(m):
        model.zero_grad()
        step_input_p, step_input_h = p_dash + k * (p - p_dash) / m, h_dash + k * (h - h_dash) / m
        step_output = model(step_input_p, step_input_h)
        step_pred = torch.argmax(step_output)
        step_grad = torch.autograd.grad(step_output[0,pred], (p, h), retain_graph=True)
        if sum_grad is None:
            sum_grad = [step_grad[0], step_grad[1]]
        else:
            sum_grad[0] += step_grad[0]
            sum_grad[1] += step_grad[1]
    sum_grad[0], sum_grad[1] = sum_grad[0] / m, sum_grad[1] / m
    sum_grad[0], sum_grad[1] = sum_grad[0] * (p - p_dash), sum_grad[1] * (h - h_dash)
    sum_grad[0], sum_grad[1] = sum_grad[0].sum(dim=2), sum_grad[1].sum(dim=2)
    relevances = [sum_grad[0].detach().cpu().numpy(), sum_grad[1].detach().cpu().numpy()]
    ptokens=[tok.text for tok in nlp.tokenizer(s1_premise)]
    htokens=[tok.text for tok in nlp.tokenizer(s2_hypothesis)]
    print("---------------------------------------------------------")
    try:
        relevances = [list(np.round(np.reshape(relevances[0],len(ptokens)),3)), list(np.round(np.reshape(relevances[1],len(htokens)),3))]
        df1 = pd.DataFrame(index=['Premise','IntegGrad'], columns=list(range(len(ptokens))), data=[ptokens, relevances[0]])
        df2 = pd.DataFrame(index=['Hypothesis','IntegGrad'], columns=list(range(len(htokens))), data=[htokens, relevances[1]])
        print("Premise : %s"%(s1_premise))
        print("Hypothesis : %s"%(s2_hypothesis))
        with pd.option_context('display.max_rows', None, 'display.max_columns', 30):
            print(df1)
            print(df2)
        print("PREDICTED Label : %s"%(answers.vocab.itos[pred]))
        return answers.vocab.itos[pred], relevances
    except:
        print("*****Error*******")
        return answers.vocab.itos[pred], []


# test queries
integrated_gradients("A black race car starts up in front of a crowd of people.","A man is driving down a lonely road.")
integrated_gradients("A soccer game with multiple males playing.","Some men are playing a sport.")
integrated_gradients("A smiling costumed woman is holding an umbrella.","A happy woman in a fairy costume holds an umbrella.")
integrated_gradients("A person on a horse jumps over a broken down airplane.","A person is training his horse for a competition.")
integrated_gradients("A person on a horse jumps over a broken down airplane.","A person is at a diner, ordering an omelette.")
integrated_gradients("A person on a horse jumps over a broken down airplane.","A person is outdoors, on a horse.")
integrated_gradients("A person on a horse jumps over a broken down airplane.","A person is indoors, on a horse.")
integrated_gradients("A person on a horse jumps over a broken down airplane.","A person is outside, on a horse.")
integrated_gradients("A person on a horse jumps over a sofa.","A person is outside, on a horse.")
integrated_gradients("A person is beside a horse.","A person is outside, on a horse.")
integrated_gradients("A person is beside a boy.","A person is outside, on a horse.")