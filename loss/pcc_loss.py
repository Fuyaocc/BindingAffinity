import torch
import torch.nn as nn

class PCCLoss(nn.Module):
    def __init__(self):
        super(PCCLoss, self).__init__()

    def forward(self, input, target):
        # 计算 PCC
        input_mean = torch.mean(input)
        target_mean = torch.mean(target)
        input_centered = input - input_mean
        target_centered = target - target_mean

        numerator = torch.sum(input_centered * target_centered)
        denominator = torch.sqrt(torch.sum(input_centered**2) * torch.sum(target_centered**2))

        pcc = numerator / (denominator + 1e-8)  # 添加一个小的常数以防分母为零

        # 将 PCC 转换为损失，可以使用 1 - PCC 作为损失，也可以使用其他形式
        loss = 1 - pcc

        return loss