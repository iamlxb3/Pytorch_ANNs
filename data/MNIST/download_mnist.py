import torchvision as ptv
import pickle
import sys
import torch as pt
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')


# train_set = ptv.datasets.MNIST("../../pytorch_database/mnist/train",train=True,transform=ptv.transforms.ToTensor(),
#                                download=True)
# test_set = ptv.datasets.MNIST("../../pytorch_database/mnist/test",train=False,transform=ptv.transforms.ToTensor(),
#                               download=True)

train_set = pickle.load(open('mnist_training_set', 'rb'))
train_set = pt.utils.data.DataLoader(train_set,batch_size=100)
print ("train_set: ", train_set)
#pickle.load(test_set, open('mnist_test_set', 'wb'))

# for data in train_set:
#     (inputs, labels) = data
#     inputs = pt.autograd.Variable(inputs).cuda()
#     print ("input: ", inputs)
#     sys.exit()
#
#     print ("labels: ", labels)

