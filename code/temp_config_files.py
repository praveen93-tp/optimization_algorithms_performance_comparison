import os
import pickle as pkl
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import seaborn as sns
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

import optimizers

optim_dict = {
    'padam':{
        'label': 'PADAM',
         'lr': 1e-3,
        'weight_decay': 1e-4
    },
    'amsgrad':{
        'label': 'Adam+AMSGrad',
        'lr': 1e-3,
        'amsgrad':True
    },
    'adabound':{
            'label': 'AdaBound',
            'lr': 1e-3,
            'amsbound':True
        },
    'adabound_w':{
                'label': 'AdaBoundW',
                'lr': 1e-3,
                'amsbound':True
            },
    'adamax': {
        'label': 'AdaMax',
        'lr': 1e-3,
    },
    'sgd': {
        'label': 'SGD',
        'lr': 1e-3
    },
    'sgd_momentum': {
        'label': 'SGD w/ momentum',
        'lr': 1e-3,
        'mu': 0.99
    },
    'sgd_nesterov': {
        'label': 'SGD w/ Nesterov momentum',
        'lr': 1e-3,
        'mu': 0.99,
        'nesterov': True
    },
    'sgd_weight_decay': {
        'label': 'SGDW',
        'lr': 1e-3,
        'mu': 0.99,
        'weight_decay': 1e-6
    },
    'sgd_lrd': {
        'label': 'SGD w/ momentum + LRD',
        'lr': 1e-3,
        'mu': 0.99,
        'lrd': 0.5
    },
    'adam': {
        'label': 'Adam',
        'lr': 1e-3
    },
    'adamW': {
        'label': 'AdamW',
        'lr': 1e-3,
        'weight_decay': 1e-4
    },
    'adam_l2': {
        'label': 'AdamL2',
        'lr': 1e-3,
        'l2_reg': 1e-4
    },
    'adam_lrd': {
        'label': 'Adam w/ LRD',
        'lr': 1e-3,
        'lrd': 0.5
    },
    'Radam': {
        'label': 'RAdam',
        'lr': 1e-3,
        'rectified': True
    },
    'RadamW': {
        'label': 'RAdamW',
        'lr': 1e-3,
        'rectified': True,
        'weight_decay': 1e-4
    },
    'Radam_lrd': {
        'label': 'RAdam w/ LRD',
        'lr': 1e-3,
        'rectified': True,
        'lrd': 0.5
    },
    'nadam': {
        'label': 'Nadam',
        'lr': 1e-3,
        'nesterov': True
    },
    'rmsprop': {
        'label': 'RMSprop',
        'lr': 1e-3,
        'beta2': 0.9,
    },
    'lookahead_sgd': {
        'label': 'Lookahead (SGD)',
        'lr': 1e-3,
        'mu': 0.99
    },
    'lookahead_adam': {
        'label': 'Lookahead (Adam)',
        'lr': 1e-3
    },
    'gradnoise_adam': {
        'label': 'Gradient Noise (Adam)',
        'lr': 1e-3
    },
    'graddropout_adam': {
        'label': 'Gradient Dropout (Adam)',
        'lr': 1e-3
    },
}


def split_optim_dict(d: dict) -> tuple:
    temp_d = deepcopy(d)
    label = temp_d['label']
    del temp_d['label']
    return label, temp_d


def load_cifar(num_train=50000, num_val=2048):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_dataset, _ = torch.utils.data.random_split(train_dataset, lengths=[num_train, len(train_dataset) - num_train])
    val_dataset, _ = torch.utils.data.random_split(val_dataset, lengths=[num_val, len(val_dataset) - num_val])

    return train_dataset, val_dataset

def load_gen(num_train=50000, num_val=2048):
    x = torch.Tensor([[0.333, 1]])
    y = torch.Tensor([[0.4, 0.2]])
    return x,y

def load_mnist(filename='data/mnist.npz', num_train=4096, num_val=512):
    data = np.load(filename)
    x_train = data['x_train'][:num_train].astype('float32')
    y_train = data['y_train'][:num_train].astype('int32')
    x_valid = data['x_test'][:num_val].astype('float32')
    y_valid = data['y_test'][:num_val].astype('int32')
    train_dataset = Dataset(x_train, y_train)
    val_dataset = Dataset(x_valid, y_valid)
    return train_dataset, val_dataset


def task_to_optimizer(task: str) -> torch.optim.Optimizer:
    optimizer = None
    if 'sgd' in task.lower():
        optimizer = getattr(optimizers, 'SGD')
    if 'adam' in task.lower():
        optimizer = getattr(optimizers, 'Adam')
    if 'rmsprop' in task.lower():
        optimizer = getattr(optimizers, 'RMSProp')
    if 'padam' in task.lower():
        optimizer = getattr(optimizers, 'Padam')
    if 'amsgrad' in task.lower():
        optimizer = getattr(optimizers, 'AMSGrad')
    if 'adabound' in task.lower():
            optimizer = getattr(optimizers, 'AdaBound')
    if 'adamax' in task.lower():
        optimizer = getattr(optimizers, 'Adamax')
    if 'adabound_w' in task.lower():
            optimizer = getattr(optimizers, 'AdaBoundW')
    if optimizer is None:
        raise ValueError(f'Optimizer for task \'{task}\' was not recognized!')
    return optimizer


def wrap_optimizer(task: str, optimizer):
    if 'gradnoise' in task.lower():
        optimizer = optimizers.GradNoise(optimizer, eta=0.3, gamma=0.55)
    if 'graddropout' in task.lower():
        optimizer = optimizers.GradDropout(optimizer, grad_retain=0.9)
    if 'lookahead' in task.lower():
        optimizer = optimizers.Lookahead(optimizer, k=5, alpha=0.5)
    return optimizer


class AvgLoss():
    def __init__(self):
        self.sum, self.avg, self.n = 0, 0, 0
        self.losses = []
    def __iadd__(self, other):
        try:
            loss = other.data.numpy()
        except:
            loss = other
        if isinstance(other, list):
            self.losses.extend(other)
            self.sum += np.sum(other)
            self.n += len(other)
        else:
            self.losses.append(float(loss))
            self.sum += loss
            self.n += 1
        self.avg = self.sum / self.n
        return self

    def __str__(self):
        return '{0:.4f}'.format(round(self.avg, 4))
    def __len__(self):
        return len(self.losses)


class Dataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def save_losses(losses, dataset: str, filename: str):
    if not os.path.exists(f'losses_{dataset}/'): os.makedirs(f'losses_{dataset}/')
    with open(f'losses_{dataset}/{filename}.pkl', 'wb') as f:
        pkl.dump(losses, f, protocol=pkl.HIGHEST_PROTOCOL)


def load_losses(dataset: str, filename: str):
    try:
        with open(f'losses_{dataset}/{filename}.pkl', 'rb') as f:
            return pkl.load(f)
    except:
        return None


def smooth(signal, kernel_size, polyorder=3):
    return savgol_filter(signal, kernel_size, polyorder)
