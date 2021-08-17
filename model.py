import torchvision
import torch.nn as nn


def get_model():

    model = torchvision.models.segmentation.fcn_resnet50(True)

    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1,1), stride=(1,1))

    return model

