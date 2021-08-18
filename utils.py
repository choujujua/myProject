import torch
import torchvision

import numpy as np

class ToTensor(object):

    def __call__(self, sample):

        img = sample['image']
        mask = sample['label']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img, 'label': mask}