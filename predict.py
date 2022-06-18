import torch
import sys
from os import system
from dataset import *
from codemaps import *
import evaluator
from utils import config
from utils.dataset import Drugdataset
from torch.utils.data import DataLoader


def output_interactions(data, sids, preds, outfile):

   #print(testdata[0])
   outf = open(outfile, 'w')
   for sid,tag in zip(sids, preds):
      exmp = data[sid]
      e1 = exmp['e1']
      e2 = exmp['e2']
      if tag!='null' :
         print(sid, e1, e2, tag, sep="|", file=outf)
            
   outf.close()


def evaluation(datadir,outfile) :
   evaluator.evaluate("DDI", datadir, outfile)


if __name__ == '__main__':
   modelname = config['modelname']
   test_path = 'devel.pck'
   testdata = Dataset(test_path)
   model = torch.load(modelname)
   codes = Codemaps(modelname)
   outfile = 'devel.out'

   X = codes.encode_words(testdata)
   Y, sids = codes.encode_labels(testdata)
   testset = Drugdataset(X, Y)

   test_size = len(testset)
   test_loader = DataLoader(testset, batch_size=test_size, shuffle=False)

   for word_data, pos_data, etype_data, data_lens, _ in test_loader:
      out = model(word_data, pos_data, etype_data, data_lens).numpy()
      Y = [codes.idx2label(np.argmax(s)) for s in Y]

# extract relations
datadir = ''
output_interactions(testdata, sids,  Y, outfile)
evaluation(datadir, outfile)




