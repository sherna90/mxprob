import numpy as np
from collections import Iterable

def one_hot(y,num_classes):
    encoding=np.zeros((len(y),num_classes))
    for i, val in enumerate(y):
        encoding[i, np.int(val)] = 1.0
    return encoding

def scaler_fit(X):
    min_col=np.amin(X,0)
    max_col=np.amax(X,0)
    X_s=(X-min_col)/(max_col-min_col)
    return X_s,min_col,max_col

def scaler_scale(X,min_col,max_col):
    X_s=(X-min_col)/(max_col-min_col)
    return X_s

def flatten(items):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

