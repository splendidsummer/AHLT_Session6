import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import math, time, copy, sys, random
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score


def set_seed(seed):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	cuda = True if torch.cuda.is_available() else False
	if cuda:
		torch.cuda.manual_seed(seed)


def convert_time():
	timestr = datetime.now()
	timestr = str(timestr)[:-9]

	ints = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
	out = '_'
	for item in timestr:
		if item in ints:
			out += item

	return out


def acc(output, labels):
	preds = output.max(1)[1].type_as(labels)
	print(preds.shape)
	correct = preds.eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)


def get_f1_score(probs, tags, data_lens, mode="macro"):
	preds = torch.argmax(probs, dim=-1)
	f1 = 0.0
	for i in range(preds.shape[0]):
		seq_len = data_lens[i]
		pred = preds[i][:seq_len]
		tag = tags[i][:seq_len]
		f1 += f1_score(pred, tag, average=mode)

	f1 = f1 / preds.shape[0]

	return f1


def get_ddi_f1_score(probs, tags, mode="macro"):
	preds = torch.argmax(probs, dim=-1)
	return f1_score(preds, tags, average=mode)


def get_max_len(parse_data):
	max_len = 0
	for lst in parse_data.data:
		sent = lst['sent']
		if len(sent) > max_len:
			max_len = len(sent)
	return max_len

#
# averages = ["micro", "macro" ]
# results = {}
# for average in averages:
# 	results[average] = f1_score(Y, Y_, average=average)


if __name__ == '__main__':
	a = torch.tensor([[[0.1, 0.5, 0.4], [0.8, 0.1, 0.1], [0.1, 0.5, 0.4], [0.8, 0.1, 0.1]],
	                  [[0.1, 0.5, 0.4], [0.8, 0.1, 0.1], [0.1, 0.5, 0.4], [0.8, 0.1, 0.1]]])

	ids = torch.argmax(a, dim=-1)
	tag = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 0]])
	b = torch.tensor([4, 3])
	f1 = get_f1_score(a, tag, b)
	print(f1)
