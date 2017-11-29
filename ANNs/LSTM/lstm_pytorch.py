import torch
import torch.nn as nn
from torch.autograd import Variable

class LstmPytorch(nn.Module):
    # INCOMPLETE
    # TODO http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    def __init__(self, input_size, hidden_size, output_size):
        super(LstmPytorch, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size + hidden_size, hidden_size)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size)).cuda()

