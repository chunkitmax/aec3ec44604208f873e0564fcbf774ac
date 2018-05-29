import torch as T


class LSTM_Model(T.nn.Module):
  def __init__(self, wordict_size, embedding_len, max_doc_len, weight=None,
               hidden_layer_size=32, num_hidden_layer=3, use_cuda=True):
    super(LSTM_Model, self).__init__()
    self.wordict_size = wordict_size
    self.embedding_len = embedding_len
    self.max_doc_len = max_doc_len
    self.weight = weight
    self.hidden_layer_size = hidden_layer_size
    self.num_hidden_layer = num_hidden_layer
    self.use_cuda = use_cuda
    self._build_model()

  def _build_model(self):
    self._build_model_1()
    self._loss_fn = T.nn.CrossEntropyLoss(weight=self.weight)
    if self.use_cuda:
      self.cuda()
    self._optimizer = T.optim.Adam(self.parameters(), 1e-3)

class CNN_Model(T.nn.Module):
  def __init__(self, wordict_size, embedding_len, max_doc_len, task,
               lr=1e-3, weight=None, use_cuda=True):
    super(CNN_Model, self).__init__()
    self.wordict_size = wordict_size
    self.embedding_len = embedding_len
    self.max_doc_len = max_doc_len
    self.task = task
    self.lr = lr
    self.weight = weight
    self.use_cuda = use_cuda
    self._build_model()

  def _build_model(self):
    self._build_model_1()
    self._loss_fn = T.nn.CrossEntropyLoss(weight=self.weight) if self.task == 'oc' \
                    else T.nn.MSELoss()
    if self.use_cuda:
      self.cuda()
    self._optimizer = T.optim.Adam(self.parameters(), self.lr)

  def forward(self, inputs):
    embeddings = self.embed(inputs)
    embeddings = T.transpose(embeddings, 1, 2)
    output = self._forward_1(embeddings)
    if self.task == 'oc':
      return output, T.max(output, dim=1)[1]
    else:
      return output, output

  def _build_model_1(self, kernel_size=(2, 3, 4), num_conv=2, num_kernel=100):
    self.embed = T.nn.Embedding(self.wordict_size, self.embedding_len, padding_idx=0,
                                scale_grad_by_freq=True)
    self.sequential = T.nn.ModuleList()
    for n in kernel_size:
      self.conv = T.nn.ModuleList()
      self.conv.append(T.nn.Conv1d(self.embedding_len, num_kernel, n, padding=n-1))
      self.conv.append(T.nn.BatchNorm1d(self.embedding_len))
      for _ in range(num_conv-1):
        self.conv.append(T.nn.Conv1d(num_kernel, num_kernel, n, padding=n-1))
        self.conv.append(T.nn.BatchNorm1d(num_kernel))
      self.sequential.append(self.conv)
    self.sequential.append(T.nn.Linear(len(kernel_size)*num_kernel, 4 if self.task == 'oc' else 1))
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
      else:
        return module(self.dropout(T.cat(output, dim=1)))

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def loss_fn(self):
    return self._loss_fn
