import torch
from torch.utils.data import Dataset, DataLoader
import os, pickle, pdb
import nltk
from xml.dom.minidom import parse


class Drugdataset(Dataset):
	def __init__(self, data, labels, data_lens):  # tag_index can be defined in other place
		super(Drugdataset, self).__init__()
		self.word_data = data[1]
		self.pos_data = data[-2]
		self.etype_data = data[-1]
		self.labels = labels
		self.data_lens = data_lens

	def __getitem__(self, index):
		word = self.word_data[index]
		pos = self.pos_data[index]
		etype = self.etype_data[index]
		label = self.labels[index]
		data_len = self.data_lens[index]

		return word, pos, etype, data_len, label

	def __len__(self):
		return len(self.labels)

	def batch_data_pro(self, batch_datas):
		(word_data, pos_data, etype_data, data_lens, batch_labels) = batch_datas
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		word_data = torch.tensor(word_data, dtype=torch.long, device=device)
		pos_data = torch.tensor(pos_data, dtype=torch.long,  device=device)
		etype_data = torch.LongTensor(etype_data, dtype=torch.long, device=device)
		batch_labels = torch.LongTensor(batch_labels, dtype=torch.long,  device=device)
		data_lens = torch.LongTensor(data_lens, dtype=torch.long, device=device)

		return word_data, pos_data, etype_data, data_lens, batch_labels


if __name__ == '__main__':
	print(111)
	#
	# train_data = pickle.load(open(parse_train_file, 'rb'))
	# devel_data = pickle.load(open(parse_devel_file, 'rb'))
	# test_data = pickle.load(open(parse_test_file, 'rb'))
	#
	# dataset = Drugdataset(train_data, word_index)
	# batch_size = 32
	# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.batch_data_pro)
	# i = 0
	#
	# for data, tag, lens in dataloader:
	# 	print('{}th data bacth'.format(i))
	#
	# 	try:
	# 		print(data.shape)
	# 		print(tag.shape)
	# 		print(lens[0])


		# except ValueError:
		# 	print('Wrong')
