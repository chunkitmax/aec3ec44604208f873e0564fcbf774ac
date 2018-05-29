import os
import pickle

from data import SemEval_DataSet
from model import CNN_Model, T
from train import Trainer


def get_dataset(task):
  if os.path.exists('data/train_%s_set'%(task)):
    train_set = pickle.load(open('data/train_%s_set'%(task), 'rb'))
  else:
    train_set = SemEval_DataSet('train', task, save=True)
    pickle.dump(train_set, open('data/train_%s_set'%(task), 'wb+'))
  if os.path.exists('data/valid_%s_set'%(task)):
    valid_set = pickle.load(open('data/valid_%s_set'%(task), 'rb'))
  else:
    valid_set = SemEval_DataSet('dev', task, wordict=train_set.wordict)
    pickle.dump(valid_set, open('data/valid_%s_set'%(task), 'wb+'))
  print('Wordict size: %s'%([len(train_set.wordict[emotion])
                             for emotion in train_set.wordict]))
  return train_set, valid_set

def CNN_OC():
  train_set, valid_set = get_dataset('oc')
  model_generator = lambda wordict_size, weight: \
                    CNN_Model(wordict_size, 100, 100, 'oc', weight)
  def collate_fn(entry):
    return entry[0], entry[1].long().unsqueeze(1)
  trainer = Trainer(model_generator, train_set, valid_set, max_epoch=750,
                    use_cuda=True, collate_fn=collate_fn)
  trainer.train()

def CNN_REG():
  train_set, valid_set = get_dataset('reg')
  model_generator = lambda wordict_size, weight: \
                    CNN_Model(wordict_size, 100, 100, 'reg', weight)
  def collate_fn(entry):
    return entry[0], entry[1].float().unsqueeze(1)
  trainer = Trainer(model_generator, train_set, valid_set, max_epoch=750,
                    use_cuda=True, collate_fn=collate_fn)
  trainer.train()

if __name__ == '__main__':
  CNN_REG()
