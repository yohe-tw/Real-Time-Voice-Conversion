import torch
import torch.nn as nn
import torch.nn.functional as F
class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super(SobelEdgeLoss, self).__init__()
        # Define Sobel kernels
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, pred, target):
        # Move Sobel kernels to the same device as the input
        sobel_x = self.sobel_x.to(pred.device)
        sobel_y = self.sobel_y.to(pred.device)

        # Compute Sobel edges for the predicted output
        pred_edges_x = torch.nn.functional.conv2d(pred, sobel_x, padding=1)
        pred_edges_y = torch.nn.functional.conv2d(pred, sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x ** 2 + pred_edges_y ** 2)

        # Compute Sobel edges for the target
        target_edges_x = torch.nn.functional.conv2d(target, sobel_x, padding=1)
        target_edges_y = torch.nn.functional.conv2d(target, sobel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x ** 2 + target_edges_y ** 2)

        # Compute the loss between edge maps
        return F.mse_loss(pred_edges, target_edges)