import torch
from torch import nn
import torch.nn.functional as F


class BaseBiLSTM(nn.Module):   # 100, 64
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


class BiLSTM(nn.Module):
	def __init__(self, num_classes, word_vocab_size,
	             type_size, pos_size, num_lstm_layers,
	             embedding_dim1, embedding_dim2, embedding_dim3,
	             hidden_dim, dropout,
	             attention_type=None, attention=False,
	             bi=True):

		super(BiLSTM, self).__init__()

		self.embedding1 = nn.Embedding(word_vocab_size, embedding_dim1, padding_idx=0)
		self.embed_dropout1 = nn.Dropout(dropout)
		self.embedding2 = nn.Embedding(pos_size, embedding_dim2, padding_idx=0)
		self.embed_dropout2 = nn.Dropout(dropout)
		self.embedding3 = nn.Embedding(type_size, embedding_dim3, padding_idx=0)
		self.embed_dropout = nn.Dropout(dropout)

		# dimension after concatenating all the embeddings
		embedding_dim = embedding_dim1 + embedding_dim2 + embedding_dim3
		# embedding_dim = embedding_dim1 + embedding_dim2
		# embedding_dim = embedding_dim1

		# convolution before LSTM
		self.conv2 = nn.Conv1d(embedding_dim, 30, 3, 1, padding='same')  # kernel_size = 3
		self.maxpool1 = nn.MaxPool1d(2)
		self.globalpool = nn.AvgPool1d(2)

		# convolution before LSTM
		self.conv3 = nn.Conv1d(hidden_dim, 30, 2, 1, padding='same')
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
		                    num_layers=num_lstm_layers, bidirectional=bi)

		self.attention = attention
		self.attention_type = attention_type

		if bi:
			hidden_dim = 2 * hidden_dim

		if self.attention != 'self' and self.attention == True:
			self.w_omega = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
			self.u_omega = nn.Parameter(torch.Tensor(hidden_dim, 1))
			nn.init.uniform_(self.w_omega, -0.1, 0.1)
			nn.init.uniform_(self.u_omega, -0.1, 0.1)

		self.classifier = nn.Linear(hidden_dim, num_classes)

		self.dropout = nn.Dropout(dropout)
		self.conv2 = nn.Conv1d(hidden_dim, 30, 2, 1, padding='same')

		# The fully-connected layer takes in the hidden dim of the LSTM and
		#  outputs a a 3x1 vector of the class scores.
		self.fc = nn.Linear(hidden_dim, num_classes)

	def soft_attention_net(self, x, query, mask=None):
		d_k = query.size(-1)  # d_k == last dim of query

		# query:[batch, seq_len, hidden_dim], x.t:[batch, hidden_dim, seq_len]
		# scores: [batch, seq_len, seq_len]
		scores = torch.matmul(query, x.transpose(1, 2)) / torch.sqrt(d_k)

		# Normalize the last dimension and get the probs
		# scores: [batch, seq_len, seq_len]
		alpha_n = F.softmax(scores, dim=-1)

		# Weighted sum of  contextual vectors,
		# [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2]
		# (after suming over the second dim) -> [batch, hidden_dim*2]
		context = torch.matmul(alpha_n, x).sum(1)

		return context

	def forward(self, x):  # we can add one more hidden input,  init hidden state somewhere
		"""
		The forward method takes in the input and the previous hidden state
		"""

		embs = self.embedding(x)
		out, _ = self.lstm(embs)

		# Dropout is applied to the output and fed to the FC layer
		out = self.dropout(out)
		out = self.fc(out)

		# We extract the scores for the final hidden state since it is the one that matters.
		out = out[:, -1]  # 这里获取的不应该是最后一个，而是根据长度获取倒数第一个输出
		return out


class BiLSTM(nn.Module):
	def __init__(self, num_classes, word_vocab_size,
	             type_size, pos_size, num_lstm_layers,
	             embedding_dim1, embedding_dim2, embedding_dim3,
	             hidden_dim, dropout,
	             attention_type=None, attention=False,
	             bi=True):

		super(BiLSTM, self).__init__()

		self.embedding1 = nn.Embedding(word_vocab_size, embedding_dim1, padding_idx=0)
		self.embed_dropout1 = nn.Dropout(dropout)
		self.embedding2 = nn.Embedding(pos_size, embedding_dim2, padding_idx=0)
		self.embed_dropout2 = nn.Dropout(dropout)
		self.embedding3 = nn.Embedding(type_size, embedding_dim3, padding_idx=0)
		self.embed_dropout = nn.Dropout(dropout)

		# dimension after concatenating all the embeddings
		embedding_dim = embedding_dim1 + embedding_dim2 + embedding_dim3
		# embedding_dim = embedding_dim1 + embedding_dim2
		# embedding_dim = embedding_dim1

		# convolution before LSTM
		self.conv2 = nn.Conv1d(embedding_dim, 30, 3, 1, padding='same')  # kernel_size = 3
		self.maxpool1 = nn.MaxPool1d(2)
		self.globalpool = nn.AvgPool1d(2)

		# convolution before LSTM
		self.conv3 = nn.Conv1d(hidden_dim, 30, 2, 1, padding='same')
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
		                    num_layers=num_lstm_layers, bidirectional=bi)

		self.attention = attention
		self.attention_type = attention_type

		if bi:
			hidden_dim = 2 * hidden_dim

		if self.attention != 'self' and self.attention == True:
			self.w_omega = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
			self.u_omega = nn.Parameter(torch.Tensor(hidden_dim, 1))
			nn.init.uniform_(self.w_omega, -0.1, 0.1)
			nn.init.uniform_(self.u_omega, -0.1, 0.1)

		self.classifier = nn.Linear(hidden_dim, num_classes)

		self.dropout = nn.Dropout(dropout)
		self.conv2 = nn.Conv1d(hidden_dim, 30, 2, 1, padding='same')

		# The fully-connected layer takes in the hidden dim of the LSTM and
		#  outputs a a 3x1 vector of the class scores.
		self.fc = nn.Linear(hidden_dim, num_classes)

	def soft_attention_net(self, x, query, mask=None):
		d_k = query.size(-1)  # d_k == last dim of query

		# query:[batch, seq_len, hidden_dim], x.t:[batch, hidden_dim, seq_len]
		# scores: [batch, seq_len, seq_len]
		scores = torch.matmul(query, x.transpose(1, 2)) / torch.sqrt(d_k)

		# Normalize the last dimension and get the probs
		# scores: [batch, seq_len, seq_len]
		alpha_n = F.softmax(scores, dim=-1)

		# Weighted sum of  contextual vectors,
		# [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2]
		# (after suming over the second dim) -> [batch, hidden_dim*2]
		context = torch.matmul(alpha_n, x).sum(1)

		return context

	def forward(self, x):  # we can add one more hidden input,  init hidden state somewhere
		"""
		The forward method takes in the input and the previous hidden state
		"""

		embs = self.embedding(x)
		out, _ = self.lstm(embs)

		# Dropout is applied to the output and fed to the FC layer
		out = self.dropout(out)
		out = self.fc(out)

		# We extract the scores for the final hidden state since it is the one that matters.
		out = out[:, -1]  # 这里获取的不应该是最后一个，而是根据长度获取倒数第一个输出
		return out


#
# # def init_hidden(self):
# # 	return (torch.zeros(1, batch_size, 32), torch.zeros(1, batch_size, 32))
#
# def forward(self, text, text_len):
# 	text_emb = self.embedding(text)
#
# 	packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
# 	packed_output, _ = self.lstm(packed_input)
# 	output, _ = pad_packed_sequence(packed_output, batch_first=True)
#
# 	out_forward = output[range(len(output)), text_len - 1, :self.dimension]
# 	out_reverse = output[:, 0, self.dimension:]
# 	out_reduced = torch.cat((out_forward, out_reverse), 1)
# 	text_fea = self.drop(out_reduced)
#
# 	text_fea = self.fc(text_fea)
# 	text_fea = torch.squeeze(text_fea, 1)
# 	text_out = torch.sigmoid(text_fea)
#

class BiLSTM_Attention(nn.Module):

	def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
		super(BiLSTM_Attention, self).__init__()

		self.hidden_dim = hidden_dim
		self.n_layers = n_layers
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
		                   bidirectional=True, dropout=0.5)

		self.fc = nn.Linear(hidden_dim * 2, 1)
		self.dropout = nn.Dropout(0.5)

	# x: [batch, seq_len, hidden_dim*2]
	# query : [batch, seq_len, hidden_dim * 2]
	# 软注意力机制 (key=value=x)
	def attention_net(self, x, query, mask=None):
		d_k = query.size(-1)  # d_k为query的维度

		# query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
		#         print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
		# 打分机制 scores: [batch, seq_len, seq_len]
		scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
		#         print("score: ", scores.shape)  # torch.Size([128, 38, 38])

		# 对最后一个维度 归一化得分
		alpha_n = F.softmax(scores, dim=-1)
		#         print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
		# 对权重化的x求和
		# [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
		context = torch.matmul(alpha_n, x).sum(1)

		return context, alpha_n

	def forward(self, x):
		# [seq_len, batch, embedding_dim]
		embedding = self.dropout(self.embedding(x))

		# output:[seq_len, batch, hidden_dim*2]
		# hidden/cell:[n_layers*2, batch, hidden_dim]
		output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
		output = output.permute(1, 0, 2)  # [batch, seq_len, hidden_dim*2]

		query = self.dropout(output)
		# 加入attention机制
		attn_output, alpha_n = self.attention_net(output, query)

		logit = self.fc(attn_output)

		return logit


if __name__ == '__main__':
	# Save and Load Functions

	def save_checkpoint(save_path, model, optimizer, valid_loss):

		if save_path == None:
			return

		state_dict = {'model_state_dict': model.state_dict(),
		              'optimizer_state_dict': optimizer.state_dict(),
		              'valid_loss': valid_loss}

		torch.save(state_dict, save_path)
		print(f'Model saved to ==> {save_path}')


	def load_checkpoint(load_path, model, optimizer):

		if load_path == None:
			return

		state_dict = torch.load(load_path, map_location=device)
		print(f'Model loaded from <== {load_path}')

		model.load_state_dict(state_dict['model_state_dict'])
		optimizer.load_state_dict(state_dict['optimizer_state_dict'])

		return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
	if save_path == None:
		return

	state_dict = {'train_loss_list': train_loss_list,
	              'valid_loss_list': valid_loss_list,
	              'global_steps_list': global_steps_list}

	torch.save(state_dict, save_path)
	print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
	if load_path == None:
		return

	state_dict = torch.load(load_path, map_location=device)
	print(f'Model loaded from <== {load_path}')

	return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']
