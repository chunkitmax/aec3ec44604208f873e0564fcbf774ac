import os
from data import SemEval_DataSet
from model import CNN_Model, T
from train import Trainer
import pickle

if __name__ == '__main__':
  if os.path.exists('data/train_set'):
    train_set = pickle.load(open('data/train_set', 'rb'))
  else:
    train_set = SemEval_DataSet('train', ('oc'), save=True)
    pickle.dump(train_set, open('data/train_set', 'wb+'))
  if os.path.exists('data/valid_set'):
    valid_set = pickle.load(open('data/valid_set', 'rb'))
  else:
    valid_set = SemEval_DataSet('dev', ('oc'))
    pickle.dump(valid_set, open('data/valid_set', 'wb+'))
  print('Wordict size: %d'%(len(train_set.wordict)))
  train_loader = T.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
  valid_loader = T.utils.data.DataLoader(valid_set, batch_size=10, shuffle=True)
  model = CNN_Model(len(train_set.wordict), 100)
  def collate_fn(entry):
    return entry[0], entry[1].unsqueeze(1)
  trainer = Trainer(model, train_loader, valid_loader, use_cuda=True, collate_fn=collate_fn)
  trainer.train()
