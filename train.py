import torch
import torch.nn as nn
from torchvision import transforms as T
from model import get_model
import pandas as pd
from data_reader import TianChiDataset
from utils import ToTensor
import torch.utils.data as D
import numpy as np

# loader data set
trfm = T.Compose([
    ToTensor()
])


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
valid_loader = D.DataLoader(valid_set, batch_size=4, shuffle=True, num_workers=2)

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

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)


# 验证数据集
def evalidate(model, valid_loader, loss_fn):

    losses = []

    model.eval()
    for i, (img, target) in enumerate(valid_loader):
        output = model(img)
        loss = loss_fcn(target, output)
        losses.append(loss.item())

    return np.array(losses).mean()




    return 0


# 训练数据集
for epoch in range(0, Epochs):

    losses = []
    for i, (img, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # print('img.shape = ', img.shape)
        target = target.long()
        output = model(img)
        loss = loss_fcn(output, target)
        print('loss = ', loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print('losses = ', np.array(losses).mean())

    # validate for two epoch

    if epoch % 2 == 0:
        eval_loss = evalidate(model, valid_set)

    # save model


    # print information when validating

    print('[epoch: {}] [loss: {}]', epoch, eval_loss)