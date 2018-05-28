import os
import pickle
import re
from collections import Counter
from zipfile import ZipFile

import numpy as np
from autocorrect import spell
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer(reduce_len=True,
                           strip_handles=True)
stemmer = PorterStemmer()
word_checker = re.compile(r'[a-zA-Z]+')

def tokenize(s):
  tokens = tokenizer.tokenize(s)
  tmp_tokens = []
  for token in tokens:
    if token.isupper() and len(token) > 1:
      tmp_tokens.append('<upper>')
    tmp_tokens.append(token)
  tokens = [stemmer.stem(spell(token))
            if word_checker.search(token) is not None
            else token
            for token in tmp_tokens]
  return tokens

def load_data(file_name, max_num_lines=None):
  tweet = {}
  affect = {}
  intensity = {}
  intensity_getter = re.compile(r'^[0-9\.\-]+')

  if max_num_lines is not None:
    max_num_lines += 1

  with ZipFile(file_name, 'r') as zf:
    file_list = zf.namelist()
    total_file_count = len(file_list)
    for index, file_name in enumerate(file_list):
      print('  file: %d / %d'%(index+1, total_file_count))
      emotion = file_name.split('-')[-2]
      tweet[emotion] = []
      affect[emotion] = []
      intensity[emotion] = []
      with zf.open(file_name, 'r') as f:
        lines = f.read().decode('utf8').strip().split('\n')[1:max_num_lines]
        total_line_count = len(lines)
        for line_no, line in enumerate(lines):
          _, _tweet, _affect, _intensity = line.strip().split('\t')
          tweet[emotion].append(tokenize(_tweet))
          affect[emotion].append(_affect)
          _intensity = float(intensity_getter.search(_intensity)[0])
          intensity[emotion].append(_intensity)
          print('\r    line: %d / %d'%(line_no+1, total_line_count), end='\033[K')
        print()
  return tweet, affect, intensity

def build_dict(data_dict, name, save=False, save_counter=False, preserved_words=()):
  if not os.path.exists('data'):
    os.mkdir('data')
  if os.path.exists('data/'+name+'_dict'):
    return pickle.load(open('data/'+name+'_dict', 'rb'))
  else:
    ret_dict = {}
    for emotion, data in data_dict.items():
      counter = Counter()
      if len(data) > 0 and \
          (isinstance(data[0], list) or \
          isinstance(data[0], tuple)):
        for entry in data:
          counter += Counter(entry)
      else:
        counter = Counter(data)
      if save_counter:
        pickle.dump(counter, open('data/'+name+'_counter', 'wb+'))
      freq_list = dict(counter.most_common())
      tmp_dict = {word: index+len(preserved_words)
                  for index, (word, _) in enumerate(freq_list.items())}
      tmp_dict.update({preserved_word: index
                       for preserved_word, index in zip(preserved_words,
                                                        range(len(preserved_words)))})
      ret_dict[emotion] = tmp_dict
    if save:
      pickle.dump(ret_dict, open('data/'+name+'_dict', 'wb+'))
    return ret_dict
