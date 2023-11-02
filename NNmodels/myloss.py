import torch
import torch.nn as nn
import torch.nn.functional as F


class SignLoss(nn.Module):
    def __init__(self, alpha=0, b=None):
        super(SignLoss, self).__init__()
        self.alpha = alpha
        self.loss = 0
        self.acc = 0

    def forward(self, args, watermark_sign, weight):
        weight_sum = torch.sum(weight, dim=1)
        out_channels = weight_sum.size()[0]
        weight_sign = torch.sign(torch.rand(out_channels) - 0.5).to(args.device)
        for i in range(out_channels):
            weight_sign[i] = weight_sum[i][0][0]


        return 0.1 * torch.sum(F.relu(-watermark_sign.view(-1) * weight_sign.view(-1) + 0.1))
