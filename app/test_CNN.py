
import io
import os
import pickle
import sys
from torch.autograd import Variable
import numpy as np
import torch as pt

#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(current_dir)
data_dir = os.path.join(top_dir, 'data')
sys.path.append(top_dir)


from ANNs.CNN.cnn_pytorch import CNNPytorch
from general_funcs.evaluate import accuracy_calculate

# config
BATCH_SIZE = 100
MAX_TRAINING_EPOCH = 3
#

# read dataset
training_data_path = os.path.join(data_dir, 'MNIST', 'mnist_training_set')
test_data_path = os.path.join(data_dir, 'MNIST', 'mnist_test_set')
training_data = pickle.load(open(training_data_path, 'rb'))
test_data = pickle.load(open(test_data_path, 'rb'))

# read into tensor
training_data = pt.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE)
test_data = pt.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
#

# build cnn
img_size = (1,1,28,28) # set the image size manually
cnn1 = CNNPytorch(img_size)
print (cnn1)
if pt.cuda.device_count() > 1:
  print("Let's use", pt.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = pt.nn.DataParallel(cnn1)
else:
    print ("cuda device number: ", pt.cuda.device_count())

if pt.cuda.is_available():
    cnn1.cuda()
#

# loss func and optim
optimizer = pt.optim.SGD(cnn1.parameters(),lr=0.01,momentum=0.9)
lossfunc = pt.nn.CrossEntropyLoss().cuda()
#

# # TEST CNN
# input = Variable(pt.randn(1, 1, 28, 28)).cuda()
# out = cnn1(input)
# print(out)
# sys.exit()
# #

# ----------------------------------------------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------------------------------------------
for epoch in range(MAX_TRAINING_EPOCH):

    for i, data in enumerate(training_data):

        optimizer.zero_grad()

        (inputs, true_labels) = data

        inputs = pt.autograd.Variable(inputs).cuda()
        true_labels = pt.autograd.Variable(true_labels).cuda()

        outputs = cnn1(inputs) # -> return pt.nn.functional.softmax(self.fc3(dout))

        # # see grads
        # for parameter in mlp1.parameters():
        #     print ("grad: ", parameter.grad)
        # sys.exit()
        # #

        loss = lossfunc(outputs, true_labels)
        loss.backward() # -> accumulates the gradient (by addition) for each parameter


        optimizer.step() # -> update weights and biases

        if i % 100 == 0:
            outputs = outputs.cpu().data.numpy()
            true_labels = true_labels.cpu().data.numpy()
            pred_labels = [np.argmax(x) for x in outputs]
            accuracy = accuracy_calculate(pred_labels, true_labels)
            print("epoch: {}, batch: {}, accuracy: {}".format(epoch, i, accuracy))
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# testing
# ----------------------------------------------------------------------------------------------------------------------
accuracy_list = []
for i, data in enumerate(test_data):
    (inputs, true_labels) = data
    inputs = pt.autograd.Variable(inputs).cuda()
    true_labels = pt.autograd.Variable(true_labels).cuda()
    outputs = cnn1(inputs)

    outputs = outputs.cpu().data.numpy()
    true_labels = true_labels.cpu().data.numpy()
    pred_labels = [np.argmax(x) for x in outputs]
    accuracy = accuracy_calculate(pred_labels, true_labels)
    accuracy_list.append(accuracy)

print ("avg_accuracy: ", sum(accuracy_list)/len(accuracy_list))
# ----------------------------------------------------------------------------------------------------------------------