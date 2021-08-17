import torch.utils.data.dataset as D
from torchvision import transforms as T
import cv2
import os
import pandas as pd

from read_rle import rle_decode
from utils import ToTensor


import matplotlib.pyplot as plt

class TianChiDataset(D.Dataset):

    def __init__(self, paths, rles, transform, train=True):
        super(TianChiDataset).__init__()

        self.paths = paths
        self.rles = rles
        self.transform = transform
        self.train = train
        self.len = len(paths)

        self.To_tensor = T.Compose([
            T.ToPILImage(),
            T.Resize(224),
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131,0.177,0.101]),
        ])


    def __getitem__(self, item):

        # read original train image
        print(os.path.join('G://myProject/train/', self.paths[item]))
        # breakpoint()
        image = cv2.imread(os.path.join('G://myProject/train/', self.paths[item]))

        # if train mode, return image and mask(type(list)), else return image for test
        if self.train:
            mask = rle_decode(self.rles[item])
            sample = {'image': image, 'mask': mask}
            # augments = self.transform(sample)
            return sample['image'], sample['mask']
            # return self.To_tensor(sample['image']), sample['mask']
        else:
            return self.To_tensor(image), ''


    def __len__(self):

        return self.len


if __name__ == '__main__':


    train_mask = pd.read_csv('G:/myProject/train_mask.csv', sep='\t', names=['name','mask'])
    print(train_mask['name'].values)
    print(train_mask['mask'].fillna('').values)
    # train_mask['name'] = train_mask['name'].apply(lambda x: '数据集/train/' + x)
    #
    # img = cv2.imread(train_mask['name'].iloc[0])
    # mask = rle_decode(train_mask['mask'].iloc[0])

    image_size = 256


    # loader data set
    trfm = T.Compose([
        ToTensor()
    ])

    dataset = TianChiDataset(
        train_mask['name'].values,
        train_mask['mask'].fillna('').values,
        transform=trfm,
        train=True
    )
    print(len(dataset))
    image, mask = dataset[0]

    # print('type(image)= ', image)
    # print('type(mask) = ', mask)

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(mask, cmap='gray')
    plt.subplot(122)
    plt.imshow(image)
    plt.show()