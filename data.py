import os
import re

import torch as T
import numpy as np
from torch.utils.data import Dataset

from preprocess import build_dict, load_data


class SemEval_DataSet(Dataset):
  def __init__(self, phase, set_name, save_counter=False,
               max_doc_len=100, max_num_lines=None, save=False,
               wordict=None, affectdict=None):
    self.data, self.affect, self.intensity = [], [], []
    preserved_words = ("<pad>", "<unk>")

    if phase not in ['train', 'dev', 'test']:
      raise AssertionError('Invalid phase')
    # find all dataset zip files
    file_list = os.listdir()
    file_pattern = re.compile(r'(2018-)?EI-([a-zA-Z]+)-En-([a-zA-Z]+)\.zip')
    file_info_list = list(map(file_pattern.search, file_list))
    total_zipfile_count = len(file_info_list)
    for index, file_info in enumerate(file_info_list):
      print('ZipFile: %d / %d'%(index+1, total_zipfile_count))
      if file_info is not None \
         and file_info.groups()[1] == set_name \
         and file_info.groups()[2] == phase:
        tweet, affect, intensity = load_data(file_info[0], max_num_lines=max_num_lines)
        self.data.extend(tweet)
        self.affect.extend(affect)
        self.intensity.extend(intensity)
    # get word_dict & label_dict
    if wordict is not None:
      self.wordict = build_dict(self.data, 'word_'+set_name, save, save_counter,
                                preserved_words=preserved_words)
    else:
      self.wordict = wordict
    if affectdict is not None:
      self.affectdict = build_dict(self.affect, 'affect'+set_name, save, save_counter)
    else:
      self.affectdict = affectdict
    # word2idx
    data = [[self.wordict[word] if word in self.wordict else preserved_words.index('<unk>')
             for word in entry][:max_doc_len]+[0]*(max_doc_len-len(entry))
            for entry in self.data]
    self.data = np.array(data, dtype=np.int32)
    # label2idx
    self.affect = [self.affectdict[affect] for affect in self.affect]
  def __getitem__(self, index):
    return T.LongTensor(self.data[index]), \
           T.LongTensor([self.affect[index]]), \
           T.FloatTensor([self.intensity[index]])
  def __len__(self):
    return self.data.shape[0]

if __name__ == '__main__':
  dataset = SemEval_DataSet('train', 'reg', max_num_lines=1)
  print(dataset)
  print(dataset.data)
  print(dataset.affect)
  print(dataset.intensity)
