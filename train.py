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
  def __init__(self, model_generator, train_dataset, valid_dataset, test_dataset,
               batch_size=50, max_epoch=1000, use_cuda=True, use_tensorboard=False,
               early_stopping_history_len=50, early_stopping_patience=5,
               collate_fn=None, verbose=1, save_best_model=False):
    self.logger = Logger(verbose_level=verbose)
    self.model_generator = model_generator
    self.train_dataset = train_dataset
    self.valid_dataset = valid_dataset
    self.test_dataset = test_dataset
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
    best_valid_corrcoef = {}
    best_test_corrcoef = {}
    for emotion in emotions:
      self.train_dataset.set_emotion(emotion)
      self.valid_dataset.set_emotion(emotion)
      self.test_dataset.set_emotion(emotion)
      train_loader = T.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                             shuffle=True)
      valid_loader = T.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size,
                                             shuffle=True)
      test_loader = T.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                            shuffle=True)
      model = self.model_generator(self.train_dataset.wordict_size, self.train_dataset.weight)
      best_valid_corrcoef[emotion], \
      best_test_corrcoef[emotion] = self._train(model, train_loader, valid_loader, test_loader,
                                                identity=emotion)
      del model, train_loader, valid_loader
    best_valid_corrcoef['avg'] = np.mean([best_valid_corrcoef[emotion] for emotion in emotions])
    best_test_corrcoef['avg'] = np.mean([best_test_corrcoef[emotion] for emotion in emotions])
    # self.logger.i('\n'+str(best_valid_corrcoef), True, True)
    # self.logger.i('\n'+str(best_test_corrcoef), True, True)
    return best_valid_corrcoef, best_test_corrcoef
  def _train(self, model, train_loader, valid_loader, test_loader, identity=None):
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
      last_test_corrcoef = -1.
      # early_stopping_violate_counter = 0
      epoch_index = 0
      for epoch_index in range(self.max_epoch):
        losses = 0.
        # acc = 0.
        counter = 0
        self.logger.i('[ %d / %d ] epoch:'%(epoch_index + 1, self.max_epoch), True)
        # Training
        model.train()
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
          output, predicted = model(data)
          # acc += (label.squeeze() == predicted).float().mean().data * data.size(0)
          loss = model.loss_fn(output, label.view(-1))
          model.optimizer.zero_grad()
          loss.backward()
          T.nn.utils.clip_grad_norm(model.parameters(), .25)
          model.optimizer.step()
          losses += loss.data.cpu()[0] * data.size(0)
          counter += data.size(0)
          progress = min((batch_index + 1) / total_batch_per_epoch * 20., 20.)
          self.logger.d('[%s] (%3.f%%) loss: %.4f, '%
                        ('>'*int(progress)+'-'*(20-int(progress)), progress * 5.,
                         losses / counter))
        mean_loss = losses / counter
        valid_losses = 0.
        valid_counter = 0
        # valid_acc = 0.
        # Validtion
        model.eval()
        valid_prediction = []
        valid_labels = []
        for entry in valid_loader:
          if self.collate_fn is not None:
            data, label = self.collate_fn(entry)
          else:
            data, label = entry
          valid_labels += list(label.view(-1))
          data = T.autograd.Variable(data)
          label = T.autograd.Variable(label)
          if self.use_cuda:
            data = data.cuda()
            label = label.cuda()
          output, predicted = model(data)
          valid_losses += model.loss_fn(output, label.view(-1)).data.cpu()[0] * data.size(0)
          valid_prediction += list(predicted.view(-1).data.tolist())
          # valid_acc += (label.squeeze() == predicted).float().mean().data * data.size(0)
          valid_counter += data.size(0)
        mean_val_loss = valid_losses/valid_counter
        # mean_val_acc = valid_acc/valid_counter
        corrcoef = np.corrcoef(valid_prediction, valid_labels)[0, 1]
        self.logger.d(' -- val_loss: %.4f, corrcoef: %.4f'%
                      (mean_val_loss, corrcoef),
                      reset_cursor=False)
        # Log with tensorboard
        if self.use_tensorboard:
          self.writer.add_scalar('train_loss', mean_loss, epoch_index)
          # self.writer.add_scalar('train_acc', acc / counter, epoch_index)
          self.writer.add_scalar('val_loss', mean_val_loss, epoch_index)
          # self.writer.add_scalar('val_acc', mean_val_acc, epoch_index)
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
        if corrcoef > best_corrcoef:
          best_corrcoef = corrcoef
          last_test_corrcoef = self._test(model, test_loader)
          self.logger.d(' -- test_corrcoef: %.4f'%(last_test_corrcoef),
                        reset_cursor=False)
          if self.save_best_model:
            self._save(model, epoch_index, loss_history, best_corrcoef, identity)
        self.logger.d('', True, False)
    except KeyboardInterrupt:
      self.logger.i('\n\nInterrupted', True)
    if self.use_tensorboard:
      self.writer.close()
    self.logger.i('Finish', True)
    return best_corrcoef, last_test_corrcoef
  def _test(self, model, test_loader):
    model.eval()
    test_prediction = []
    test_labels = []
    for entry in test_loader:
      if self.collate_fn is not None:
        data, label = self.collate_fn(entry)
      else:
        data, label = entry
      test_labels += list(label.view(-1))
      data = T.autograd.Variable(data)
      label = T.autograd.Variable(label)
      if self.use_cuda:
        data = data.cuda()
        label = label.cuda()
      _, predicted = model(data)
      test_prediction += list(predicted.view(-1).data.tolist())
    return np.corrcoef(test_prediction, test_labels)[0, 1]
  def _save(self, model, global_step, loss_history, best_corrcoef, identity):
    T.save({
        'epoch': global_step+1,
        'state_dict': model.state_dict(),
        'loss_history': loss_history,
        'best_corrcoef': best_corrcoef,
        'optimizer': model.optimizer.state_dict()
    }, identity+'_best')

# if __name__ == '__main__':
#   trainer = Trainer(use_cuda=True, use_tensorboard=True)
#   trainer.train()
