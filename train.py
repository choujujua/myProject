import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

import torchvision
from torchvision import transforms as T
from model import get_model
from loss import get_loss
import pandas as pd
from data_reader import TianChiDataset
from utils import ToTensor
import torch.utils.data as D

# loader data set
trfm = T.Compose([
    ToTensor()
])
print('111')

train_mask = pd.read_csv('G:/myProject/train_mask.csv', sep='\t', names=['name','mask'])
dataset = TianChiDataset(
        train_mask['name'].values,
        train_mask['mask'].fillna('').values,
        transform=trfm,
        train=True
    )

train_idx, valid_idx = [], []
for i in range(len(dataset)):
    if i % 7 == 0:
        train_idx.append(i)
    else:
        valid_idx.append(i)

train_set = D.Subset(dataset, train_idx)
valid_set = D.Subset(dataset, valid_idx)

train_loader = D.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)

Epochs = 1
batch_size = 32
image_size = 256
Device = 'cuda' if torch.cuda.is_available() else 'cpu'

def loss_fcn(pred, target):

    bce_loss = nn.CrossEntropyLoss()
    bce = bce_loss(pred, target)

    return bce

model = get_model()
# loss = get_loss()

optimiter = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

def evalidate(model, dataset):

    # losses = []



    return loss

for epoch in range(0, Epochs):

    losses = []
    for i, (img, target) in enumerate(train_loader):
        print('img.shape = ', img.shape)
        target = target.long()
        output = model(img)
        print('output.shape = ', output.shape)
        print('target.shape = ', target.shape)
        print('target = ', target)
        loss = loss_fcn(output, target)
        loss.backward()
        optimiter.step()
        losses.append(loss.item())

    # validate for two epoch

    if epoch % 2 == 0:
        eval_loss = evalidate(model, valid_set)

    # print information when validating

    print('[epoch: {}] [loss: {}]', epoch, eval_loss)