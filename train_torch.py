import debugpy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch import optim
from utils.dataset import Drugdataset
from torch.utils.data import DataLoader
import numpy as np
import math, time, copy, sys, random, pickle, json
from codemaps import *
from utils.utils import *
from utils.losses import *
from utils.dataset import Drugdataset
from utils.config import config
from models.BILSTM import BaseBiLSTM, BiLSTM_CNN
from sklearn.metrics import f1_score, accuracy_score
import wandb
import torchmetrics

if config['debug']:
	import debugpy

	debugpy.listen(("0.0.0.0", 8888))
	print("Waiting for client to attach...")
	debugpy.wait_for_client()

device = config['device']
parse_train_file = 'train.pck'
parse_devel_file = 'devel.pck'
pars_test_file = 'test.pck'

traindata = Dataset(parse_train_file)
valdata = Dataset(parse_devel_file)

max_len = 150
codes = Codemaps(traindata, max_len)

Xt = codes.encode_words(traindata)
Yt, _ = codes.encode_labels(traindata)
train_lens = codes.data_lens
Xv = codes.encode_words(valdata)
Yv, _ = codes.encode_labels(valdata)
val_lens = codes.data_lens
assert train_lens != val_lens

trainset = Drugdataset(Xt, Yt, train_lens)
print(len(trainset))
train_loader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
develset = Drugdataset(Xv, Yv, val_lens)
devel_size = len(develset)
devel_loader = DataLoader(develset, batch_size=devel_size, shuffle=False)

vocab_size = codes.get_n_lc_words()
pos_size = codes.get_n_pos()
etype_size = codes.get_n_drugtypes()
num_classes = codes.get_n_labels()
pad_idx = config['pad']

model = BaseBiLSTM(num_classes, vocab_size, pos_size, etype_size,
                   config['num_lstm_layers'], config['embedding_dim1'],
                   config['embedding_dim2'], config['embedding_dim3'],
                   config['hidden_dim'], config['dropout'], pad_idx)

model.to(device=device)

if config['optimizer'] == 'adam':
	optimizer = optim.Adam(model.parameters(), lr=config['lr'])
elif 'sgd' == config['optimizer']:
	optimizer = optim.SGD(model.parameters(), lr=config['lr'])

if config['loss'] == 'focal':
	loss_fn = FocalLoss(ignore_index=pad_idx)
else:
	loss_fn = nn.CrossEntropyLoss()


def train(epoches):
	if config['track_wandb']:
		run_name = f"BILSTM_{int(time.time())}"
		if config['track']:
			wandb.init(
				project=config.wandb_project,
				name=run_name,
				config=config,
			)

	# global devel_out
	train_losses, devel_losses, train_accs, devel_macro, devel_micro = [], [], [], [], []
	for epoch in range(epoches):
		model.train()
		train_loss = 0.0
		total_correct = 0
		total_count = 0
		for (word_data, pos_data, etype_data, data_lens, batch_labels) in train_loader:
			batch_idx = 1
			print('Epoch {}, Batch')
			optimizer.zero_grad()
			out = model(word_data, pos_data, etype_data, data_lens)
			loss = loss_fn(out, batch_labels)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
			preds = torch.argmax(out, dim=-1)
			correct = preds.eq(batch_labels.double())
			total_count += batch_labels.shape[0]
			total_correct += correct.sum()
			batch_idx += 1

		train_acc = total_correct / total_count
		train_losses.append(train_loss)
		train_accs.append(train_acc)

		model.eval()

		for word_data, pos_data, etype_data, data_lens, batch_labels in devel_loader:
			# print(devel_data.shape)
			out = model(word_data, pos_data, etype_data, data_lens)

			devel_macro_f1 = get_ddi_f1_score(out.cpu(), batch_labels.cpu())
			devel_micro_f1 = get_ddi_f1_score(out.cpu(), batch_labels.cpu(), mode='micro')
			devel_loss = loss_fn(out, batch_labels)
			devel_acc = torchmetrics.functional.accuracy(out, batch_labels)

		devel_losses.append(devel_loss.item())
		devel_macro.append(devel_macro_f1)
		devel_micro.append(devel_micro_f1)

		if config['track_wandb']:
			wandb.log(
				{"train_loss": train_loss, "acc_train": train_acc, "loss_devel": devel_loss, "acc_devel": devel_acc,
				 "macro_f1_devel": devel_macro_f1, "micro_f1_devel": devel_micro_f1})

		print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(train_loss),
		      'acc_train: {:.4f}'.format(train_acc), 'loss_devel: {:.4f}'.format(devel_loss),
		      'macro_f1_devel: {:.4f}'.format(devel_macro_f1),
		      'micro_f1_devel: {:.4f}'.format(devel_micro_f1),
		      'acc_devel: {:.4f}'.format(devel_acc))

	results = {'train_loss': train_losses, 'train_acc': train_accs, 'test_loss': devel_losses,
	           'macro_f1_devel': devel_macro, 'micro_f1_devel': devel_micro_f1}

	results_file = './results/bilstm/base.pck'
	with open(results_file, 'wb') as f:
		pickle.dump(results, f)

	torch.save(model, config['bilstm_base'])

	wandb.finish()
	with open(results_file, 'wb') as f:
		pickle.dump(results, f)

	return None


if __name__ == '__main__':
	train(config['epoches'])
	codes.save(config['bilstm_base'])
