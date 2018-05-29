import os
import re

import numpy as np
import torch as T
from torch.utils.data import Dataset

from preprocess import build_dict, load_data


class SemEval_DataSet(Dataset):
  EMOTIONS = ['joy', 'sadness', 'fear', 'anger']

  def __init__(self, phase, set_name, save_counter=False,
               max_doc_len=100, max_num_lines=None, save=False,
               wordict=None, affectdict=None):
    preserved_words = ("<pad>", "<unk>")

    if phase not in ['train', 'dev', 'test']:
      raise AssertionError('Invalid phase')
    # find all dataset zip files
    file_list = os.listdir()
    file_pattern = re.compile(r'(2018-)?EI-([a-zA-Z]+)-En-([a-zA-Z]+)\.zip')
    file_info_list = list(map(file_pattern.search, file_list))
    file_info_list = list(filter(lambda x: x is not None and
                                 x.groups()[1] == set_name and
                                 x.groups()[2] == phase, file_info_list))
    total_zipfile_count = len(file_info_list)
    for index, file_info in enumerate(file_info_list):
      print('ZipFile: %d / %d'%(index+1, total_zipfile_count))
      self.data, self.affect, self.intensity = load_data(file_info[0], max_num_lines=max_num_lines)
    # get word_dict & label_dict
    if wordict is None:
      self.wordict = build_dict(self.data, 'word_'+set_name, save, save_counter,
                                preserved_words=preserved_words)
    else:
      self.wordict = wordict
    if affectdict is None:
      self.affectdict = build_dict(self.affect, 'affect'+set_name, save, save_counter)
    else:
      self.affectdict = affectdict
    # word2idx
    tmp_data = {}
    for emotion, data in self.data.items():
      tmp_data[emotion] = np.array([[self.wordict[emotion][word]
                                     if word in self.wordict[emotion]
                                     else preserved_words.index('<unk>')
                                     for word in entry][:max_doc_len]+[0]*(max_doc_len-len(entry))
                                    for entry in data], np.int32)
    self.data = tmp_data
    # label2idx
    tmp_affect = {}
    for emotion, affect_list in self.affect.items():
      tmp_affect[emotion] = [self.affectdict[emotion][affect] for affect in affect_list]
    self.affect = tmp_affect
    self.emotion = None
  def set_emotion(self, emotion):
    assert emotion in self.EMOTIONS
    self.emotion = emotion
  def __getitem__(self, index):
    return T.LongTensor(self.data[self.emotion][index]), \
           T.LongTensor([self.affect[self.emotion][index]]), \
           self.intensity[self.emotion][index]
  @property
  def wordict_size(self):
    return len(self.wordict[self.emotion])
  def __len__(self):
    return self.data[self.emotion].shape[0]

if __name__ == '__main__':
  dataset = SemEval_DataSet('train', 'emotion', 'reg', max_num_lines=1)
  print(dataset)
  print(dataset.data)
  print(dataset.affect)
  print(dataset.intensity)
