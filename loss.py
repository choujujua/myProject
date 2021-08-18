import torch.nn as nn

def get_loss(pred, target):


    bce = nn.BCELoss(pred, target)

    return bce
