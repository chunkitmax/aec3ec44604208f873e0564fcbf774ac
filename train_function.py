import os
import pickle

from data import SemEval_DataSet
from model import T, CNN_Model, ResNet_GRU_Model, NN_Model
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
  if os.path.exists('data/test_%s_set'%(task)):
    test_set = pickle.load(open('data/test_%s_set'%(task), 'rb'))
  else:
    test_set = SemEval_DataSet('test', task, wordict=train_set.wordict)
    pickle.dump(test_set, open('data/test_%s_set'%(task), 'wb+'))
  print('Wordict size: %s'%([len(train_set.wordict[emotion])
                             for emotion in train_set.wordict]))
  return train_set, valid_set, test_set

def CNN_OC():
  train_set, valid_set, test_set = get_dataset('oc')
  def model_generator(wordict_size, weight):
    model = CNN_Model(wordict_size, 100, 100, 'oc', weight)
    model.build_model()
    return model
  def collate_fn(entry):
    return entry[0], entry[1].long().unsqueeze(1)
  trainer = Trainer(model_generator, train_set, valid_set, test_set, max_epoch=750,
                    use_cuda=True, collate_fn=collate_fn)
  print(trainer.train())

def CNN_REG():
  train_set, valid_set, test_set = get_dataset('reg')
  def model_generator(wordict_size, weight):
    model = CNN_Model(wordict_size, 100, 100, 'reg', weight)
    model.build_model(lr=5e-5)
    return model
  def collate_fn(entry):
    return entry[0], entry[1].float().unsqueeze(1)
  trainer = Trainer(model_generator, train_set, valid_set, test_set, max_epoch=750,
                    use_cuda=True, collate_fn=collate_fn, verbose=1)
  print(trainer.train())

def ResNetCNN_OC():
  train_set, valid_set, test_set = get_dataset('oc')
  hyparameter_sets = [
      {
          'lr': 5e-4,
          'num_kernel': 64,
          'kernel_size': 3,
          'hidden_layer_size': 16,
          'num_hidden_layer': 1,
          'dropout': 0.
      }
  ]
  results = []
  for hyparameter_set in hyparameter_sets:
    def model_generator(wordict_size, weight):
      model = ResNet_GRU_Model(wordict_size, 100, 60, 'oc', weight)
      model.build_model(**hyparameter_set)
      return model
    def collate_fn(entry):
      return entry[0][:, :60], entry[1].long().unsqueeze(1)
    trainer = Trainer(model_generator, train_set, valid_set, test_set, max_epoch=750,
                      use_cuda=True, collate_fn=collate_fn)
    results.append(trainer.train())
  print(results)

def ResNetCNN_REG():
  train_set, valid_set, test_set = get_dataset('reg')
  hyparameter_sets = [
      {
          'lr': 5e-5,
          'num_kernel': 64,
          'kernel_size': 3,
          'hidden_layer_size': 16,
          'num_hidden_layer': 1,
          'dropout': 0.
      }
  ]
  results = []
  for hyparameter_set in hyparameter_sets:
    def model_generator(wordict_size, weight):
      model = ResNet_GRU_Model(wordict_size, 100, 60, 'reg', weight)
      model.build_model(**hyparameter_set)
      return model
    def collate_fn(entry):
      return entry[0][:, :60], entry[1].float().unsqueeze(1)
    trainer = Trainer(model_generator, train_set, valid_set, test_set, max_epoch=750,
                      use_cuda=True, collate_fn=collate_fn)
    results.append(trainer.train())
  print(results)

def NN_OC():
  train_set, valid_set, test_set = get_dataset('oc')
  hyparameter_sets = [
      {
          'lr': 1e-3,
          'hidden_layer_size': 125,
          'num_hidden_layer': 2
      }
  ]
  results = []
  for hyparameter_set in hyparameter_sets:
    def model_generator(wordict_size, weight):
      model = NN_Model(wordict_size, 100, 100, 'oc', weight)
      model.build_model(**hyparameter_set)
      return model
    def collate_fn(entry):
      return entry[0], entry[1].long().unsqueeze(1)
    trainer = Trainer(model_generator, train_set, valid_set, test_set, max_epoch=750,
                      use_cuda=True, collate_fn=collate_fn)
    results.append(trainer.train())
  print(results)

def NN_REG():
  train_set, valid_set, test_set = get_dataset('reg')
  hyparameter_sets = [
      {
          'lr': 5e-5,
          'hidden_layer_size': 80,
          'num_hidden_layer': 12,
          'dropout': .2
      }
  ]
  results = []
  for hyparameter_set in hyparameter_sets:
    def model_generator(wordict_size, weight):
      model = NN_Model(wordict_size, 100, 100, 'reg', weight)
      model.build_model(**hyparameter_set)
      return model
    def collate_fn(entry):
      return entry[0], entry[1].float().unsqueeze(1)
    trainer = Trainer(model_generator, train_set, valid_set, test_set, max_epoch=750,
                      use_cuda=True, collate_fn=collate_fn)
    results.append(trainer.train())
  print(results)

if __name__ == '__main__':
  NN_REG()
