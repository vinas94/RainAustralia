import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from benedict import benedict
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, classification_report


class timeseries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
  
    def __len__(self):
        return self.len
    
    
class LSTM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=5, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        out, _status = self.lstm(x)
        out = out[:,-1,:] # out.shape: [128, 14, 5] - here we are taking the last day
        out = self.fc1(torch.relu(out))
        return out
    
    
def read_and_parse(df, cities=[], columns=[]):
    
    # Read in data
    out = pd.read_csv('./Data/X_'+df+'.csv')
    outy = pd.read_csv('./Data/y_'+df+'.csv')
    
    # Add the target to features
    out['y'] = outy

    # Filtering for cities
    if len(cities)>0:
        out = out.loc[out['Location'].isin(cities),:]
        
    # Separating target
    outy = out['y']

    # Dropping other columns
    if len(columns)>0:
        columns = np.concatenate([['Date', 'Location'], columns])
        out = out[columns]
    else:
        out = out.drop('y', axis=1)
    
    return out, outy


def transform_to_TS(X, y, n=14):
    
    X = X.copy()
    y = y.copy()
    
    X['Date'] = [datetime.strptime(x, '%Y-%m-%d').date() for x in X['Date']]
    X['y'] = y
    
    Xcomb = []
    ycomb = []
    
    for l in X['Location'].unique():
        
        XL = X.query('Location=="'+l+'"').sort_values('Date').reset_index(drop=True)
        
        Xloc = np.array(XL.drop(['Location', 'y'], axis=1))
        yloc = np.array(XL['y'])
    
        Xout = [] 
        Yout = [] 
        for i in range(len(Xloc)-n):
            listas = []
            for j in range(i, i+n):
                listas.append(Xloc[j])
            Xout.append(listas)
            Yout.append(yloc[j])
            
        Xcomb.extend(Xout)
        ycomb.extend(Yout)
    
    Xcomb = np.array(Xcomb)
    ycomb = np.array(ycomb)
    
    ######################
    
    idx = []
    for i in range(Xcomb.shape[0]):
        idx.append((Xcomb[i, n-1, 0] - Xcomb[i, 0, 0]).days)
        
    Xcomb = Xcomb[np.array(idx)==n-1, :, 1:]
    ycomb = ycomb[np.array(idx)==n-1]
    
    Xcomb = np.array(Xcomb, dtype=np.float32)
    ycomb = np.array(ycomb, dtype=np.float32)

    return np.array(Xcomb), np.array(ycomb)


def get_predictions(X, model=None):
    outputs = model(X)
    return torch.sigmoid(outputs).round().squeeze().detach().numpy()


def plot_conf_matrix(y_true, y_pred, cmap='viridis', normalise=None, annot=True, fmt=None, fontsize=10, ax=None):
    
    CM = confusion_matrix(y_true, y_pred)

    if normalise=='true':
        CM = CM/CM.sum(axis=1).reshape(2,1)
    elif normalise=='pred':
        CM = CM/CM.sum(axis=0)
    elif normalise=='all':
        CM = CM/CM.sum()

    if fmt==None:
        fmt = '.0f' if normalise==None else '.2f'
        
    heat = sns.heatmap(CM, cmap=cmap, annot=annot, fmt=fmt, square=True, ax=ax, annot_kws={'fontsize':fontsize})
    plt.xlabel('Predicted label', fontsize=fontsize)
    plt.ylabel('True label', fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    cbar = heat.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize)
    for _, spine in heat.spines.items():
        spine.set_visible(True)
        
        
def class_report(y_true, y_pred, reset_index=False):
    clr = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).round(2).T
    clr['support'][-3] = clr['support'][-1]
    clr['support'] = clr['support'].astype(int)
    if reset_index:
        clr = clr.reset_index()
        clr.columns = ['class', 'precision', 'recall', 'f1-score', 'support']
    return clr


def del_dir(run, directory):
    
    structure = benedict(run.get_structure())
    namespace = benedict()
    namespace['.'.join(directory.split('/'))] = structure['.'.join(directory.split('/'))]
    
    def get_dirs(run, namespace, path='', files=[]):
        for k, v in namespace.items():
            if isinstance(v, dict):
                get_dirs(run, v, path+'/'+k, files)
            else:
                files.append(path+'/'+k)

        return files
    
    files = get_dirs(run, namespace)
    for i in files:
        run.pop(i)
        
    files = []
    
    
def get_dir(run, directory):
    
    structure = benedict(run.get_structure())
    namespace = benedict()
    namespace['.'.join(directory.split('/'))] = structure['.'.join(directory.split('/'))]

    def get_dirs(run, namespace, path='', files=[]):
        for k, v in namespace.items():
            if isinstance(v, dict):
                get_dirs(run, v, path+'/'+k, files)
            else:
                files.append(path+'/'+k)

        return files

    files = get_dirs(run, namespace)
    short_files = [x[len('.'.join('params'.split('/')[:-1]))+1:] for x in files]
    d = benedict()
    for i in range(len(files)):
        try:
            try:
                d[short_files[i].split('/')[1:]] = run[files[i]].fetch()
            except:
                d[short_files[i].split('/')[1:]] = run[files[i]].fetch_values()['value'].to_numpy()
        except:
            pass
        
    files = []
    return d