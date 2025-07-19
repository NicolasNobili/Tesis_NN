import torch
import torch.nn as nn
import torch.nn.functional as F

class HistogramLoss(nn.Module):
    """
    HistogramLoss compares the color distribution between predicted and target images
    using per-channel histograms. It computes the L1 distance between normalized histograms
    of each color channel (R, G, B) and averages the result over the batch and channels.

    This loss encourages the output image to have a similar overall color distribution
    to the ground truth image, which can be useful for tasks like image super-resolution
    or style-preserving generation.

    Args:
        num_bins (int): Number of bins to use when computing the histograms. Default is 256.
                        This assumes image pixel values are in the range [0, 1].
    """
    def __init__(self, num_bins=256):
        super(HistogramLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, output, target):
        """
        Compute the histogram-based L1 loss between output and target images.

        Args:
            output (Tensor): The predicted image tensor of shape (B, 3, H, W),
                             with values in the range [0, 1].
            target (Tensor): The ground truth image tensor of shape (B, 3, H, W),
                             with values in the range [0, 1].

        Returns:
            Tensor: A scalar tensor representing the average L1 distance between
                    histograms of the predicted and target images.
        """
        batch_size, channels, height, width = output.shape
        loss = 0.0

        for b in range(batch_size):
            for c in range(channels):
                # Flatten the c-th channel of the b-th image
                out_channel = output[b, c, :, :].flatten()
                tgt_channel = target[b, c, :, :].flatten()

                # Compute histograms with specified number of bins over the [0, 1] range
                hist_out = torch.histc(out_channel, bins=self.num_bins, min=0.0, max=1.0)
                hist_tgt = torch.histc(tgt_channel, bins=self.num_bins, min=0.0, max=1.0)

                # Normalize histograms to form valid distributions
                hist_out = hist_out / hist_out.sum()
                hist_tgt = hist_tgt / hist_tgt.sum()

                # Compute L1 loss (sum of absolute differences) between histograms
                loss += F.l1_loss(hist_out, hist_tgt)

        # Average the loss over the number of histograms (batch * channels)
        loss = loss / (batch_size * channels)
        return loss
