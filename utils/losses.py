import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    '''
    Multi-class Focal loss implementation, refer to the following link:
    https://github.com/lonePatient/BERT-NER-Pytorch/blob/master/losses/focal_loss.py
    '''

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [Batch_size, Seq_len, Num_classes]
        target: [Batch_size, Seq_len]
        """
        logpt = F.log_softmax(input, dim=-1)
        print(logpt.shape)
        pt = torch.exp(logpt)
        print(pt.shape)
        logpt = (1-pt)**self.gamma * logpt
        print(logpt.shape)
        loss = F.nll_loss(logpt.view(-1, logpt.shape[-1]), target.view(-1), self.weight,ignore_index=self.ignore_index)
        return loss


if __name__ == '__main__':

    num_classes = 29
    num_sample = 10

    loss_fn = FocalLoss()

    preds = torch.randn((5, 20, 20))
    labels = torch.concat([torch.arange(20), torch.arange(20), torch.arange(20),
                           torch.arange(20), torch.arange(20)], dim=0)
    labels = torch.stack([torch.arange(20), torch.arange(20), torch.arange(20),
                           torch.arange(20), torch.arange(20)])

    loss = loss_fn(preds, labels)

    print(11)


