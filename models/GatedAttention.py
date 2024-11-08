import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    def __init__(self,n_classes):
        super(GatedAttention, self).__init__()
        # 全连接层的隐含单元
        self.L = 512 #512
        self.D = 128
        self.K = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),  #（ in_dimension, out，dimension）
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, x)  # KxL

        return M , A # Y_prob Y_hat A