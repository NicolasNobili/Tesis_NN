import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLoss(nn.Module):
    """
    Edge Loss based on gradient comparison between the super-resolved (SR) 
    and high-resolution (HR) ground truth images.
    
    Converts images to grayscale, computes gradients using Sobel filters,
    and penalizes differences in edge intensity (L1 loss between gradients).
    """
    def __init__(self):
        super(EdgeLoss, self).__init__()

        # Sobel kernel for horizontal (x-direction) edges
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)

        # Sobel kernel for vertical (y-direction) edges
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

        # Register Sobel filters as buffers (non-trainable parameters)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def rgb_to_grayscale(self, img):
        """
        Converts RGB or multi-channel image to grayscale.
        Uses standard luminance conversion for RGB.
        Averages all channels for other formats (e.g., multispectral).

        Args:
            img (Tensor): Image tensor of shape (B, C, H, W)
        
        Returns:
            Tensor: Grayscale image of shape (B, 1, H, W)
        """
        if img.shape[1] == 3:
            r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        else:
            # For multispectral or non-RGB: average across channels
            return img.mean(dim=1, keepdim=True)

    def forward(self, sr, hr):
        """
        Computes the edge-based loss between SR and HR images.

        Args:
            sr (Tensor): Super-resolved image, shape (B, C, H, W)
            hr (Tensor): High-resolution ground truth image, shape (B, C, H, W)

        Returns:
            Tensor: Scalar edge loss (L1 between gradients)
        """
        # Convert both images to grayscale
        sr_gray = self.rgb_to_grayscale(sr)
        hr_gray = self.rgb_to_grayscale(hr)

        # Compute gradient maps using Sobel filters
        grad_sr_x = F.conv2d(sr_gray, self.sobel_x, padding=1)
        grad_sr_y = F.conv2d(sr_gray, self.sobel_y, padding=1)
        grad_hr_x = F.conv2d(hr_gray, self.sobel_x, padding=1)
        grad_hr_y = F.conv2d(hr_gray, self.sobel_y, padding=1)

        # Compute L1 loss between SR and HR gradients
        loss_x = F.l1_loss(grad_sr_x, grad_hr_x)
        loss_y = F.l1_loss(grad_sr_y, grad_hr_y)

        # Return total edge loss
        return loss_x + loss_y
