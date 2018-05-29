'''
train.py
train model
'''

import gzip
import sys
from collections import deque
from zipfile import ZipFile

import numpy as np
import torch as T
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from log import Logger


class Trainer:
  def __init__(self, model=None, train_dataset=None, valid_dataset=None,
               batch_size=50, max_epoch=1000, use_cuda=True, use_tensorboard=False,
               early_stopping_history_len=50, early_stopping_patience=5,
               collate_fn=None, verbose=1, save_best_model=False):
    self.logger = Logger(verbose_level=verbose)
    self.model = model
    self.train_dataset = train_dataset
    self.valid_dataset = valid_dataset
    self.batch_size = batch_size
    self.max_epoch = max_epoch
    self.use_cuda = use_cuda
    self.use_tensorboard = use_tensorboard
    self.early_stopping_history_len = early_stopping_history_len
    self.early_stopping_patience = early_stopping_patience
    self.collate_fn = collate_fn
    self.save_best_model = save_best_model
    self.counter = 0
  def train(self):
    emotions = self.train_dataset.EMOTIONS
    best_corrcoef = {}
    for emotion in emotions:
      self.train_dataset.set_emotion(emotion)
      self.valid_dataset.set_emotion(emotion)
      train_loader = T.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                             shuffle=True)
      valid_loader = T.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size,
                                             shuffle=True)
      self.model._build_model(self.train_dataset.wordict_size)
      best_corrcoef[emotion] = self._train(train_loader, valid_loader, identity=emotion)
    best_corrcoef['avg'] = np.mean([best_corrcoef[emotion] for emotion in emotions])
    self.logger.i('\n'+str(best_corrcoef), True, True)
  def _train(self, train_loader, valid_loader, identity=None):
    if identity is None:
      identity = 'Net'+str(self.counter)
      self.counter += 1
    if self.use_tensorboard:
      from tensorboardX import SummaryWriter  
      self.writer = SummaryWriter(identity+'_logs')
    self.logger.i('Start training %s...'%(identity), True)
    try:
      total_batch_per_epoch = len(train_loader)
      loss_history = deque(maxlen=self.early_stopping_history_len)
      best_corrcoef = -1.
      # early_stopping_violate_counter = 0
      epoch_index = 0
      for epoch_index in range(self.max_epoch):
        losses = 0.
        acc = 0.
        counter = 0
        self.logger.i('[ %d / %d ] epoch:'%(epoch_index + 1, self.max_epoch), True)
        # Training
        self.model.train()
        for batch_index, entry in enumerate(train_loader):
          if self.collate_fn is not None:
            data, label = self.collate_fn(entry)
          else:
            data, label = entry
          data = T.autograd.Variable(data)
          label = T.autograd.Variable(label)
          if self.use_cuda:
            data = data.cuda()
            label = label.cuda()
          self.model.optimizer.zero_grad()
          output, predicted = self.model(data)
          acc += (label.squeeze() == predicted).float().mean().data
          loss = self.model.loss_fn(output, label.view(-1))
          loss.backward()
          self.model.optimizer.step()
          losses += loss.data.cpu()[0]
          counter += 1
          progress = min((batch_index + 1) / total_batch_per_epoch * 20., 20.)
          self.logger.d('[%s] (%3.f%%) loss: %.4f, acc: %.4f'%
                        ('>'*int(progress)+'-'*(20-int(progress)), progress * 5.,
                         losses / counter, acc / counter))
        mean_loss = losses / counter
        valid_losses = 0.
        valid_fscore = 0.
        valid_step = 0
        # Validtion
        self.model.eval()
        valid_data = []
        valid_labels = []
        for valid_step, entry in enumerate(valid_loader):
          if self.collate_fn is not None:
            data, label = self.collate_fn(entry)
          else:
            data, label = entry
          valid_labels += list(label.view(-1))
          data = T.autograd.Variable(T.LongTensor(data))
          label = T.autograd.Variable(T.LongTensor(label))
          if self.use_cuda:
            data = data.cuda()
            label = label.cuda()
          output, predicted = self.model(data)
          valid_data += list(predicted.view(-1).data.tolist())
          fscore = precision_recall_fscore_support(label.squeeze().data.tolist(),
                                                   predicted.data.tolist(),
                                                   average='weighted')[2]
          valid_fscore += fscore
          valid_losses += self.model.loss_fn(output, label.view(-1)).data.cpu()[0]
        mean_val_loss = valid_losses/(valid_step+1)
        mean_fscore = valid_fscore/(valid_step+1)
        corrcoef = np.corrcoef(valid_data, valid_labels)[0, 1]
        self.logger.d(' -- val_loss: %.4f, fscr: %.4f, corrcoef: %.4f'%
                      (mean_val_loss, mean_fscore, corrcoef),
                      reset_cursor=False)
        # Log with tensorboard
        if self.use_tensorboard:
          self.writer.add_scalar('train_loss', mean_loss, epoch_index)
          self.writer.add_scalar('train_acc', acc / counter, epoch_index)
          self.writer.add_scalar('val_loss', mean_val_loss, epoch_index)
          self.writer.add_scalar('val_fscr', mean_fscore, epoch_index)
          self.writer.add_scalar('val_corrcoef', corrcoef, epoch_index)
        loss_history.append(mean_val_loss)
        # # Early stopping
        # if mean_val_loss > np.mean(loss_history):
        #   early_stopping_violate_counter += 1
        #   if early_stopping_violate_counter >= self.early_stopping_patience:
        #     self.logger.i('Early stopping...', True)
        #     break
        # else:
        #   early_stopping_violate_counter = 0
        # Save best model
        if self.save_best_model and corrcoef > best_corrcoef:
          self._save(epoch_index, loss_history, mean_fscore, identity)
          best_corrcoef = corrcoef
        self.logger.d('', True, False)
    except KeyboardInterrupt:
      self.logger.i('\n\nInterrupted', True)
    if self.use_tensorboard:
      self.writer.close()
    self.logger.i('Finish', True)
    return best_corrcoef
  # def _test(self, test_file='test_for_you_guys.csv'):
  #   zf = ZipFile(self.target_file, 'r')
  #   raw_data = zf.read(test_file).decode("ISO-8859-1").strip().split('\n')
  #   whole_dataset = Dataset(raw_data[1:], 1.0, is_test_set=True, max_len=max_len)
  #   test_data_loader = DataLoader(whole_dataset, batch_size=self.batch_size, shuffle=False)
  #   with open('submission.csv', 'w+') as sf:
  #     sf.write('%s,sentiment\n'%(raw_data[0][:-1]))
  #     counter = 1
  #     self.model.eval()
  #     for data, _ in test_data_loader:
  #       data = T.autograd.Variable(data)
  #       if self.use_cuda:
  #         data = data.cuda()
  #       _, predicted = self.model(data)
  #       for prediction in predicted.cpu().data.tolist():
  #         sf.write('%s,%s\n'%(raw_data[counter][:-1],
  #                             [key for key, value in whole_dataset.sentiments.items()
  #                              if value == prediction][0]))
  #         counter += 1
  #   print('Finished')
  def _save(self, global_step, loss_history, best_corrcoef, identity):
    T.save({
        'epoch': global_step+1,
        'state_dict': self.model.state_dict(),
        'loss_history': loss_history,
        'best_corrcoef': best_corrcoef,
        'optimizer': self.model.optimizer.state_dict()
    }, identity+'_best')

if __name__ == '__main__':
  trainer = Trainer(use_cuda=True, use_tensorboard=True)
  trainer.train()
