import torch as T

class NN_Model(T.nn.Module):
  def __init__(self, wordict_size, embedding_len, max_doc_len, task,
               weight=None, use_cuda=True):
    super(NN_Model, self).__init__()
    self.wordict_size = wordict_size
    self.embedding_len = embedding_len
    self.max_doc_len = max_doc_len
    self.task = task
    self.weight = weight
    self.use_cuda = use_cuda
    # self.build_model()

  def build_model(self, lr=1e-3, **args):
    self._build_model_1(**args)
    self.embedding_dropout = T.nn.Dropout(0.1)
    self._loss_fn = T.nn.CrossEntropyLoss(weight=self.weight) if self.task == 'oc' \
                    else T.nn.MSELoss()
    if self.use_cuda:
      self.cuda()
    self._optimizer = T.optim.Adam(self.parameters(), lr)

  def forward(self, inputs):
    embeddings = self.embed(inputs)
    embeddings = T.sum(embeddings, 1)
    output = self._forward_1(embeddings)
    if self.task == 'oc':
      return output, T.max(output, dim=1)[1]
    return output, output

  def _build_model_1(self, hidden_layer_size=128, num_hidden_layer=3, dropout=.5):
    self.embed = T.nn.Embedding(self.wordict_size, self.embedding_len, padding_idx=0,
                                scale_grad_by_freq=True)
    self.sequential = T.nn.ModuleList()
    if num_hidden_layer == 0:
      self.sequential.append(T.nn.Linear(self.embedding_len, 4 if self.task == 'oc' else 1))
    else:
      self.sequential.append(T.nn.Linear(self.embedding_len, hidden_layer_size))
      for _ in range(num_hidden_layer-2):
        self.sequential.append(T.nn.Linear(hidden_layer_size, hidden_layer_size))
      self.sequential.append(T.nn.Linear(hidden_layer_size, 4 if self.task == 'oc' else 1))
    self.dropout = T.nn.Dropout(dropout)
    self.activation = lambda x: T.nn.functional.leaky_relu(x, 0.1)

  def _forward_1(self, embeddings):
    tmp_output = embeddings
    for module in self.sequential:
      if isinstance(module, T.nn.Linear):
        tmp_output = self.dropout(module(tmp_output))
    return tmp_output

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def loss_fn(self):
    return self._loss_fn

class ResNet_GRU_Model(T.nn.Module):
  def __init__(self, wordict_size, embedding_len, max_doc_len, task,
               weight=None, use_cuda=True):
    super(ResNet_GRU_Model, self).__init__()
    self.wordict_size = wordict_size
    self.embedding_len = embedding_len
    self.max_doc_len = max_doc_len
    self.task = task
    self.weight = weight
    self.use_cuda = use_cuda
    # self.build_model()

  def build_model(self, lr=1e-3, **args):
    self._build_model_1(**args)
    self.embedding_dropout = T.nn.Dropout(0.1)
    self._loss_fn = T.nn.CrossEntropyLoss(weight=self.weight) if self.task == 'oc' \
                    else T.nn.MSELoss()
    if self.use_cuda:
      self.cuda()
    self._optimizer = T.optim.Adam(self.parameters(), lr)

  def _build_model_1(self, num_kernel=64, kernel_size=3, hidden_layer_size=32,
                     num_hidden_layer=3, dropout=0., bidirectional=True):
    self.embed = T.nn.Embedding(self.wordict_size, self.embedding_len, padding_idx=0,
                                scale_grad_by_freq=True)
    self.sequential = T.nn.ModuleList()
    self.sequential.append(T.nn.Conv1d(self.embedding_len, num_kernel, 7, padding=3))
    for _ in range(3*2):
      self.sequential.append(T.nn.Conv1d(num_kernel, num_kernel, kernel_size, padding=kernel_size//2))
    self.sequential.append(T.nn.GRU(num_kernel, hidden_layer_size,
                                    num_hidden_layer, batch_first=True,
                                    dropout=dropout, bidirectional=bidirectional))
    self.sequential.append(T.nn.Linear(hidden_layer_size*self.max_doc_len*(2 if bidirectional else 1), 128))
    self.sequential.append(T.nn.Linear(128, 4 if self.task == 'oc' else 1))
    self.dropout = T.nn.Dropout()
    self.activation = lambda x: T.nn.functional.leaky_relu(x, 0.1)

  def forward(self, inputs):
    embeddings = self.embed(inputs)
    embeddings = self.embedding_dropout(T.transpose(embeddings, 1, 2))
    output = self._forward_1(embeddings)
    if self.task == 'oc':
      return output, T.max(output, dim=1)[1]
    return output, output

  def _forward_1(self, embeddings):
    tmp_output = embeddings
    tmp_skip_src = None
    CNN_count = 0
    for module in self.sequential:
      if isinstance(module, T.nn.Conv1d):
        if tmp_skip_src is None:
          tmp_skip_src = self.activation(module(tmp_output))
          tmp_output = tmp_skip_src
        else:
          if CNN_count % 2 == 0 and CNN_count > 0:
            tmp_output += tmp_skip_src
            tmp_output = tmp_output
          tmp_output = self.activation(module(tmp_output))
          CNN_count += 1
      elif isinstance(module, T.nn.GRU):
        tmp_output += tmp_skip_src
        tmp_output = T.transpose(tmp_output, 1, 2)
        tmp_output = module(tmp_output)
      elif isinstance(module, T.nn.Linear):
        if tmp_skip_src is not None:
          tmp_ouput = self.activation(module(self.dropout(tmp_output[0].contiguous()
                                                                       .view(embeddings.shape[0], -1))))
          tmp_skip_src = None
        else:
          return module(tmp_ouput)

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def loss_fn(self):
    return self._loss_fn

class CNN_Model(T.nn.Module):
  def __init__(self, wordict_size, embedding_len, max_doc_len, task,
               weight=None, use_cuda=True):
    super(CNN_Model, self).__init__()
    self.wordict_size = wordict_size
    self.embedding_len = embedding_len
    self.max_doc_len = max_doc_len
    self.task = task
    self.weight = weight
    self.use_cuda = use_cuda
    # self.build_model()

  def build_model(self, lr=1e-3, **args):
    self._build_model_1(**args)
    self.embedding_dropout = T.nn.Dropout(0.1)
    self._loss_fn = T.nn.CrossEntropyLoss(weight=self.weight) if self.task == 'oc' \
                    else T.nn.MSELoss()
    if self.use_cuda:
      self.cuda()
    self._optimizer = T.optim.Adam(self.parameters(), lr)

  def forward(self, inputs):
    embeddings = self.embed(inputs)
    embeddings = self.embedding_dropout(T.transpose(embeddings, 1, 2))
    output = self._forward_1(embeddings)
    if self.task == 'oc':
      return output, T.max(output, dim=1)[1]
    return output, output

  def _build_model_1(self, kernel_size=(2, 3, 4), num_conv=2, num_kernel=100):
    self.embed = T.nn.Embedding(self.wordict_size, self.embedding_len, padding_idx=0,
                                scale_grad_by_freq=True)
    self.sequential = T.nn.ModuleList()
    for n in kernel_size:
      self.conv = T.nn.ModuleList()
      self.conv.append(T.nn.Conv1d(self.embedding_len, num_kernel, n, padding=n-1))
      for _ in range(num_conv-1):
        self.conv.append(T.nn.Conv1d(num_kernel, num_kernel, n, padding=n-1))
      self.sequential.append(self.conv)
    self.sequential.append(T.nn.Linear(len(kernel_size)*num_kernel, 128))
    self.sequential.append(T.nn.Linear(128, 4 if self.task == 'oc' else 1))
    self.dropout = T.nn.Dropout()
    self.activation = lambda x: T.nn.functional.leaky_relu(x, 0.1)
    self.max_pool = lambda x: T.nn.functional.max_pool1d(x, x.shape[2])
  def _forward_1(self, embeddings):
    output = []
    for module in self.sequential:
      if isinstance(module, T.nn.ModuleList):
        tmp_output = embeddings
        for layer in module:
          if isinstance(layer, T.nn.Conv1d):
            tmp_output = self.activation(layer(tmp_output))
          else:
            tmp_output = layer(tmp_output)
        output.append(self.max_pool(tmp_output).squeeze(2))
      elif output is not None:
        tmp_output = self.activation(module(self.dropout(T.cat(output, dim=1))))
        output = None
      else:
        tmp_output = module(tmp_output)
    return tmp_output

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def loss_fn(self):
    return self._loss_fn
