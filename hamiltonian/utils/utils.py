import numpy as np
from collections import Iterable
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_diagnostics(diagnostics,title,file_name):
    labels, data = diagnostics.keys(), diagnostics.values()
    flatten_data=list()
    for d in data:
        flatten_data.append(d.reshape(-1))
    plt.rcParams['figure.dpi'] = 360
    fig=plt.figure(figsize=[5,5])
    sns.set_style("whitegrid")    
    plt.boxplot(flatten_data)
    plt.xticks(range(1, len(labels) + 1), labels,rotation=70)
    plt.title(title)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def plot_loss(loss,title,file_name):
    plt.rcParams['figure.dpi'] = 360
    fig=plt.figure(figsize=[5,5])
    sns.set_style("whitegrid")
    plt.plot(loss[0],color='blue',lw=3)
    plt.plot(loss[1],color='red',lw=3)
    plt.xlabel('Epoch', size=18)
    plt.ylabel('Loss', size=18)
    plt.title(title, size=18)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()