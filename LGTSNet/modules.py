import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Set, List



"""Model Parts"""

class SC(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, padding: Union[int, tuple] = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv1d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        padding=self.padding, groups=self.c_in)
        self.conv1d_1x1 = nn.Conv1d(self.c_in, self.c_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        y = self.depthwise_conv(x)
        y = self.conv1d_1x1(y)
        return y