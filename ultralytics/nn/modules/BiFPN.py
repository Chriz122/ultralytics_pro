import torch.nn as nn
import torch
 
class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
 
class Bi_FPN(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.swish = swish()
        self.epsilon = 0.0001
 
    def forward(self, x):
        weights = self.weight / (torch.sum(self.swish(self.weight), dim=0) + self.epsilon) # 权重归一化处理
        weighted_feature_maps = [weights[i] * x[i] for i in range(len(x))]
        stacked_feature_maps = torch.stack(weighted_feature_maps, dim=0)
        result = torch.sum(stacked_feature_maps, dim=0)
        return result


class Concat_BiFPN(nn.Module):
    def __init__(self, dimension=1):
        super(Concat_BiFPN, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
 
    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)