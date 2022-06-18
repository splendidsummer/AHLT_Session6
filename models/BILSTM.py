import torch
from torch import nn
import torch.nn.functional as F


class BaseBiLSTM(nn.Module):   # 100, 64
	def __init__(self, num_classes, vocab_size,
	             pos_size, drugtype_size, num_lstm_layers,
	             embedding_dim1, embedding_dim2, embedding_dim3,
	             hidden_dim, dropout, bi=True):
		super(BaseBiLSTM, self).__init__()

		self.embedding1 = nn.Embedding(vocab_size, embedding_dim1, padding_idx=0)
		self.embed_dropout1 = nn.Dropout(dropout)
		self.embedding2 = nn.Embedding(pos_size, embedding_dim2, padding_idx=0)
		self.embed_dropout2 = nn.Dropout(dropout)
		self.embedding3 = nn.Embedding(drugtype_size, embedding_dim3, padding_idx=0)
		self.embed_dropout3 = nn.Dropout(dropout)

		# dimension after concatenating all the embeddings
		embedding_dim = embedding_dim1 + embedding_dim2 + embedding_dim3
		# embedding_dim = embedding_dim1 + embedding_dim2
		# embedding_dim = embedding_dim1

		self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
		                    num_layers=num_lstm_layers, dropout=dropout,
		                    bidirectional=True)

		hidden_dim = 2 * hidden_dim

		self.classifier = nn.Linear(hidden_dim, num_classes)

	def forward(self, xw, xp, xdt, lens):  # we can add one more hidden input,  init hidden state somewhere
		"""
		The forward method takes in the input and the previous hidden state
		"""
		embW = self.embed_dropout1(self.embedding1(xw))
		embP = self.embed_dropout2(self.embedding2(xp))
		embDT = self.embed_dropout3(self.embedding3(xdt))

		embs = torch.cat([embW, embP, embDT], dim=-1)

		pack = nn.utils.rnn.pack_padded_sequence(embs, lens, batch_first=True, enforce_sorted=False)
		out, (h, c) = self.lstm(pack)  # out is the hidden state, _ is memory state
		out, lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
		# Get the last output to do the predict
		out = F.relu(out[:,-1])
		out = self.classifier(out)

		return out


class BiLSTM_CNN(nn.Module):   # 100, 64
	def __init__(self, num_classes, word_vocab_size,
	             pos_size, drug_type_size, num_lstm_layers,
	             embedding_dim1, embedding_dim2, embedding_dim3,
	             hidden_dim, dropout, bi=True):
		super(BaseBiLSTM, self).__init__()

		self.embedding1 = nn.Embedding(word_vocab_size, embedding_dim1, padding_idx=0)
		self.embed_dropout1 = nn.Dropout(dropout)
		self.embedding2 = nn.Embedding(pos_size, embedding_dim2, padding_idx=0)
		self.embed_dropout2 = nn.Dropout(dropout)
		self.embedding3 = nn.Embedding(drug_type_size, embedding_dim3, padding_idx=0)
		self.embed_dropout3 = nn.Dropout(dropout)

		# dimension after concatenating all the embeddings
		embedding_dim = embedding_dim1 + embedding_dim2 + embedding_dim3
		# embedding_dim = embedding_dim1 + embedding_dim2
		# embedding_dim = embedding_dim1

		self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
		                    num_layers=num_lstm_layers, dropout=dropout,
		                    bidirectional=bi)

		if bi:
			hidden_dim = 2 * hidden_dim

		self.classifier = nn.Linear(hidden_dim, num_classes)

	def forward(self, xw, xp, xdt, data_len):  # we can add one more hidden input,  init hidden state somewhere
		"""
		The forward method takes in the input and the previous hidden state
		"""
		embW = self.embed_dropout1(self.embedding1(xw))
		embP = self.embed_dropout2(self.embedding2(xp))
		embDT = self.embed_dropout3(self.embedding3(xdt))

		embs = torch.cat([embW, embP, embDT], dim=-1)
		pack = nn.utils.rnn.pack_padded_sequence(embs, data_len, batch_first=True, enforce_sorted=False)
		out, _ = self.lstm(pack)  # out is the hidden state, _ is memory state
		out, lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

		# Get the last output to do the predict
		out = F.relu(out[:,-1])
		out = self.fc(out)

		return out


class CNNBILSTM(nn.Module):   # 100, 64
	def __init__(self, num_classes, word_vocab_size,
	             pos_size, drug_type_size, num_lstm_layers,
	             embedding_dim1, embedding_dim2, embedding_dim3,
	             hidden_dim, dropout, bi=True):
		super(CNNBILSTM, self).__init__()

		self.embedding1 = nn.Embedding(word_vocab_size, embedding_dim1, padding_idx=0)
		self.embed_dropout1 = nn.Dropout(dropout)
		self.embedding2 = nn.Embedding(pos_size, embedding_dim2, padding_idx=0)
		self.embed_dropout2 = nn.Dropout(dropout)
		self.embedding3 = nn.Embedding(drug_type_size, embedding_dim3, padding_idx=0)
		self.embed_dropout3 = nn.Dropout(dropout)

		# dimension after concatenating all the embeddings
		embedding_dim = embedding_dim1 + embedding_dim2 + embedding_dim3
		# embedding_dim = embedding_dim1 + embedding_dim2
		# embedding_dim = embedding_dim1

		self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
		                    num_layers=num_lstm_layers, dropout=dropout,
		                    bidirectional=bi)

		if bi:
			hidden_dim = 2 * hidden_dim

		self.classifier = nn.Linear(hidden_dim, num_classes)

	def forward(self, xw, xp, xdt, data_len):  # we can add one more hidden input,  init hidden state somewhere
		"""
		The forward method takes in the input and the previous hidden state
		"""
		embW = self.embed_dropout1(self.embedding1(xw))
		embP = self.embed_dropout2(self.embedding2(xp))
		embDT = self.embed_dropout3(self.embedding3(xdt))

		embs = torch.cat([embW, embP, embDT], dim=-1)
		pack = nn.utils.rnn.pack_padded_sequence(embs, data_len, batch_first=True, enforce_sorted=False)
		out, _ = self.lstm(pack)  # out is the hidden state, _ is memory state
		out, lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

		# Get the last output to do the predict
		out = F.relu(out[:,-1])
		out = self.fc(out)

		return out

