import sys
import numpy as np
from sklearn.metrics import accuracy_score

def accuracy_calculate(pred, true):
    accuracy = accuracy_score(true, pred)
    return accuracy

