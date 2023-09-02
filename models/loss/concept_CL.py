import torch
import torch.nn as nn
import torch.nn.functional as F

# Check in 2023-06-12
class CL(nn.Module):
    def __init__(self):
        super(CL, self).__init__()
        self.temperature = 1

    def forward(self, feature_1: torch.Tensor, feature_2: torch.Tensor):
        feature_1_norm = F.normalize(feature_1, dim=1) 
        feature_2_norm = F.normalize(feature_2, dim=1)

        similarity_matrix = torch.matmul(feature_1_norm, feature_2_norm.T) / self.temperature
        similarity_matrix = torch.exp(similarity_matrix)
        similarity_row_sum = torch.sum(similarity_matrix, dim=1)

        loss = -torch.log(torch.diag(similarity_matrix) / similarity_row_sum)
        return loss.mean()