import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, batch_dice=False, square = False, smooth=1.):
        super(SoftDiceLoss, self).__init__()

        self.batch_dice = batch_dice
        self.smooth = smooth
        self.square = square

    def forward(self, x, y_onehot):

        if self.batch_dice:
            axes = [0] + list(range(2, len(x.shape)))
        else:
            axes = list(range(2, len(x.shape)))

        intersect = x * y_onehot
        
        if self.square:

            denominator = x ** 2 + y_onehot ** 2

        else:
            denominator = x  + y_onehot 


        intersect = intersect.sum(axes) 

        denominator = denominator.sum(axes) 


        dc = ((2 * intersect) + self.smooth) / (denominator + self.smooth)

        dc = dc.mean()


        return 1-dc

class PseudoScalarLoss(nn.Module):
    def __init__(self):
        super(PseudoScalarLoss, self).__init__()

    def forward(self, x, y):

        #absy = abs(y)
        #loss = absy * torch.log(1.0 + torch.exp(-x * y))
        #loss = loss + (1.0 - absy) * torch.square(x)
        #loss = loss.mean()
        #return loss

        #labels = torch.unique(y)
        labels = [-1,0,1]
        ##print("labels = ", labels)
        loss  = 0
        #for l in enumerate(labels):
        for l in labels:
            if l == 0:
                loss += torch.sum((x[:,0,...][y==l]) **2 )
            elif l < 0:
                loss += torch.sum(torch.log(1+torch.exp(x[:,0,...][y==l])))
            elif l > 0:
                loss += torch.sum(torch.log(1+torch.exp(-x[:,0,...][y==l])))

        return loss / (x.shape[-1]*x.shape[-2]*x.shape[-3])
