import torch
from utils.utils import *

config = {}

config['track_wandb'] = False
config['wandb_project'] = 'AHLT'

config['sort_train'] = False

pad_idx = 0

embedding_dim1 = 100
embedding_dim2 = 20
embedding_dim3 = 10

config['debug'] = False
config['pad'] = 0
config['lr'] = 1e-4
config['drug_vocab'] = '../data/preprocess/drug_vocab.pkl'
config['drug_suf_vocab'] = '../data/preprocess/drug_suf_vocab.pkl'
config['batch_size'] = 32
config['epoches'] = 2
config['cuda'] = True if torch.cuda.is_available() else False
config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config['seed'] = 168
config['weight_decay'] = 1e-5
config['dropout'] = 0.5
config['train_data_file'] = '../data/preprocess/parse_train_data.pkl'
config['devel_data_file'] = '../data/preprocess/parse_devel_data.pkl'
config['test_data_file'] = '../data/preprocess/parse_test_data.pkl'
config['model'] = 'bilstm'  # 'bilstm', 'bilstm_crf', 'transformers', 'bert_lstm_crf'
config['model_path'] = './models/' + config['model'] + '/' + convert_time() + '/' + '.pkl'
config['glove_embed_file'] = '../data/glove/embedding_mat.pkl'
config['optimizer'] = 'adam'
config['loss'] = 'ce'  # 'ce' or 'focal'
config['eval'] = 'devel'
config['num_lstm_layers'] = 1
config['embedding_dim1'] = 50
config['embedding_dim2'] = 10
config['embedding_dim3'] = 4
config['hidden_dim'] = 128
config['pad'] = 0

# BILSTM model params
config['bilstm_embedding_dim'] = 128
config['bilstm_hidden_dim'] = 128
config['bilstm_bi'] = True
config['num_layers'] = 1
config['bilstm_use_pretrain'] = False

config['bilstm_base'] = './save_model/bilstm/base.pt'



# BILSTM_CRF params
config['bcrf_lr'] = 1e-4


# BERT_LSTM_CRF params


# Transformers
