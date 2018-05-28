import torch as T


class LSTM_Model(T.nn.Module):
  def __init__(self, ):
    super(LSTM_Model, self).__init__()

  def _build_model(self):
    pass

class CNN_Model(T.nn.Module):
  def __init__(self, wordict_size, max_doc_len, use_cuda=True):
    super(CNN_Model, self).__init__()
    self.wordict_size = wordict_size
    self.max_doc_len = max_doc_len
    self.use_cuda = use_cuda
    self._build_model()

  def _build_model(self):
    self.embed = T.nn.Embedding(self.wordict_size, 50, padding_idx=0, scale_grad_by_freq=True)
    self.conv = T.nn.Conv1d(50, 32, 3)
    self.pool = T.nn.MaxPool1d(self.max_doc_len-2)
    self.dense = T.nn.Linear(16, 4)

    self._loss_fn = T.nn.CrossEntropyLoss()
    if self.use_cuda:
      self.cuda()
    self._optimizer = T.optim.Adam(self.parameters(), 1e-3)

  def forward(self, inputs):
    embeddings = self.embed(inputs)
    conv = self.conv(T.transpose(embeddings, 1, 2))
    pool = self.pool(conv)
    dense1 = self.dense1(pool.squeeze())
    output = self.dense2(self.dropout(dense1))
    return output, T.max(output, dim=1)[1]

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def loss_fn(self):
    return self._loss_fn
