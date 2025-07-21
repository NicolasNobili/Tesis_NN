import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLossRGB(nn.Module):
    """
    Edge Loss por canal: calcula gradientes con filtros Sobel por separado
    en cada canal (R, G, B) y compara con L1 los gradientes de la imagen SR
    contra la HR.
    """
    def __init__(self):
        super(EdgeLossRGB, self).__init__()

        # Sobel kernels (3x3) para X e Y
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_gradients(self, img):
        """
        Aplica filtros Sobel a cada canal por separado.
        
        Args:
            img: Tensor de forma (B, C, H, W)
        
        Returns:
            grad_x, grad_y: Gradientes por canal (B, C, H, W)
        """
        grads_x = []
        grads_y = []

        for c in range(img.shape[1]):
            channel = img[:, c:c+1, :, :]  # (B, 1, H, W)
            grad_x = F.conv2d(channel, self.sobel_x, padding=1)
            grad_y = F.conv2d(channel, self.sobel_y, padding=1)
            grads_x.append(grad_x)
            grads_y.append(grad_y)

        # Reconstruir tensor (B, C, H, W)
        grad_x = torch.cat(grads_x, dim=1)
        grad_y = torch.cat(grads_y, dim=1)

        return grad_x, grad_y

    def forward(self, sr, hr):
        """
        Calcula la pérdida L1 entre los gradientes por canal.
        """
        grad_sr_x, grad_sr_y = self.compute_gradients(sr)
        grad_hr_x, grad_hr_y = self.compute_gradients(hr)

        loss_x = F.l1_loss(grad_sr_x, grad_hr_x)
        loss_y = F.l1_loss(grad_sr_y, grad_hr_y)

        return loss_x + loss_y