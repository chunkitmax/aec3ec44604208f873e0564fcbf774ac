import argparse
import os
import pickle

from data import SemEval_DataSet
from model import CNN_Model, T
from train import Trainer

parser = argparse.ArgumentParser(description='ELEC4010I/COMP4901I')
parser.add_argument('task', type=str, help='Task [oc/reg]')
parser.add_argument('phase', type=str, help='Phase [train/test]')

parser.add_argument('-b', '--batch_size', default=10, type=int, help='Batch size')
parser.add_argument('-emb', '--emb_len', default=50, type=int, help='Embedding length')
parser.add_argument('-ml', '--max_len', default=100, type=int, help='Max document length')

parser.add_argument('-tb', '--tensorboard', action='store_true',
                    help='Gen log files for Tensorboard')
parser.add_argument('-s', '--save', action='store_true', help='Save best model')

Args = parser.parse_args()

if __name__ == '__main__':
  # Checking
  if Args.task in ['oc', 'reg'] and Args.phase in ['train', 'test']:
    # Loading data set
    if os.path.exists('data/train_%s_set'%(Args.task)):
      train_set = pickle.load(open('data/train_%s_set'%(Args.task), 'rb'))
    else:
      train_set = SemEval_DataSet('train', 'oc', save=True)
      pickle.dump(train_set, open('data/train_%s_set'%(Args.task), 'wb+'))
    if os.path.exists('data/valid_%s_set'%(Args.task)):
      valid_set = pickle.load(open('data/valid_%s_set'%(Args.task), 'rb'))
    else:
      valid_set = SemEval_DataSet('dev', 'oc', wordict=train_set.wordict,
                                  affectdict=train_set.affectdict)
      pickle.dump(valid_set, open('data/valid_%s_set'%(Args.task), 'wb+'))
    print('Wordict size: %d'%(len(train_set.wordict)))
    if Args.phase == 'train':
      # Training
      model = CNN_Model(Args.emb_len, Args.max_len)
      def collate_fn(entry):
        return entry[0], entry[1].long().unsqueeze(1)
      trainer = Trainer(model, train_set, valid_set, use_cuda=True, collate_fn=collate_fn,
                        use_tensorboard=Args.tensorboard, save_best_model=Args.save)
      trainer.train()
    else:
      # Testing
      pass
  else:
    parser.print_help()
