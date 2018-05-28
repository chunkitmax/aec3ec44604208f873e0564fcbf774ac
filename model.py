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

  def _build_model_1(self):
    self.embed = T.nn.Embedding(self.wordict_size, self.embedding_len, padding_idx=0,
                                scale_grad_by_freq=True)
    self.conv = T.nn.Conv1d(self.embedding_len, 200, 3, padding=2)
    self.conv_1 = T.nn.Conv1d(200, 200, 3, padding=2)
    self.conv_2 = T.nn.Conv1d(200, 200, 3, padding=2)
    self.conv2 = T.nn.Conv1d(self.embedding_len, 200, 4, padding=3)
    self.conv2_1 = T.nn.Conv1d(200, 200, 4, padding=3)
    self.conv2_2 = T.nn.Conv1d(200, 200, 4, padding=3)
    self.conv3 = T.nn.Conv1d(self.embedding_len, 200, 5, padding=4)
    self.conv3_1 = T.nn.Conv1d(200, 200, 5, padding=4)
    self.conv3_2 = T.nn.Conv1d(200, 200, 5, padding=4)
    self.pool = T.nn.MaxPool1d(self.max_doc_len)
    self.pool2 = T.nn.MaxPool1d(self.max_doc_len)
    self.pool3 = T.nn.MaxPool1d(self.max_doc_len)
    self.dropout = T.nn.Dropout()
    self.dense = T.nn.Linear(200*3, 4)
  def _forward_1(self, embeddings):
    conv = self.pool(self.conv_2(self.conv_2(self.conv(embeddings))))
    conv2 = self.pool2(self.conv2_2(self.conv2_1(self.conv2(embeddings))))
    conv3 = self.pool3(self.conv3_2(self.conv3_1(self.conv3(embeddings))))
    concat = T.cat([conv.squeeze(), conv2.squeeze(), conv3.squeeze()], dim=1)
    return self.dense(concat)

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def loss_fn(self):
    return self._loss_fn
