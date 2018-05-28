import torch as T


class LSTM_Model(T.nn.Module):
  def __init__(self, ):
    super(LSTM_Model, self).__init__()

  def _build_model(self):
    pass

class CNN_Model(T.nn.Module):
  def __init__(self, wordict_size, embedding_len, max_doc_len, use_cuda=True):
    super(CNN_Model, self).__init__()
    self.wordict_size = wordict_size
    self.embedding_len = embedding_len
    self.max_doc_len = max_doc_len
    self.use_cuda = use_cuda
    self._build_model()

  def _build_model(self):
    self._build_model_1()
    self._loss_fn = T.nn.CrossEntropyLoss()
    if self.use_cuda:
      self.cuda()
    self._optimizer = T.optim.Adam(self.parameters(), 3e-3)

  def forward(self, inputs):
    embeddings = self.embed(inputs)
    embeddings = T.transpose(embeddings, 1, 2)
    output = self._forward_1(embeddings)
    return output, T.max(output, dim=1)[1]

  def _build_model_1(self, kernel_size=(3, 4, 5), num_conv=3, num_kernel=200):
    self.embed = T.nn.Embedding(self.wordict_size, self.embedding_len, padding_idx=0,
                                scale_grad_by_freq=True)
    self.sequential = T.nn.ModuleList()
    for n in kernel_size:
      self.conv = T.nn.ModuleList()
      self.conv.append(T.nn.Conv1d(self.embedding_len, num_kernel, n, padding=n-1))
      for _ in range(num_conv-1):
        self.conv.append(T.nn.Conv1d(num_kernel, num_kernel, n, padding=n-1))
      self.sequential.append(self.conv)
    self.sequential.append(T.nn.Linear(len(kernel_size)*num_kernel, 4))
    self.dropout = T.nn.Dropout()
    self.activation = lambda x: T.nn.functional.leaky_relu(x, 0.1)
    self.max_pool = lambda x: T.nn.functional.max_pool1d(x, x.shape[2])
  def _forward_1(self, embeddings):
    output = []
    for module in self.sequential:
      if isinstance(module, T.nn.ModuleList):
        tmp_output = embeddings
        for layer in module:
          tmp_output = self.activation(layer(tmp_output))
        output.append(self.max_pool(tmp_output).squeeze())
      else:
        return module(T.cat(output, dim=1))

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def loss_fn(self):
    return self._loss_fn
