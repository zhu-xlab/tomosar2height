import torch.nn as nn


class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.

    Args:
        size_in (int): Input dimension.
        size_out (int): Output dimension. Defaults to `size_in` if not specified.
        size_h (int): Hidden dimension. Defaults to the smaller of `size_in` and `size_out` if not specified.
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()

        # Default values for size_out and size_h
        size_out = size_out if size_out is not None else size_in
        size_h = size_h if size_h is not None else min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Layers
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        # Shortcut connection
        self.shortcut = nn.Linear(size_in, size_out, bias=False) if size_in != size_out else None

        # Initialize the second layer weights to zero for residual learning
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        """
        Forward pass through the ResNet block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, size_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, size_out).
        """
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
