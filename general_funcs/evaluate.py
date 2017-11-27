import sys
import numpy as np
from sklearn.metrics import accuracy_score

def accuracy_calculate(pred, true):
    accuracy = accuracy_score(true, pred)
    return accuracy

# accuarcy
def AccuarcyCompute(pred,label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
#     print(pred.shape(),label.shape())
    test_np = (np.argmax(pred,1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)