import torch
import torch.nn as nn
from core.layers.cross_attention import *

class ConvGRUCell(nn.Module):
    def __init__(self, input_channel, hidden_channel,   width , kernel_size):
        super(ConvGRUCell, self).__init__()

        padding = kernel_size // 2  # 保持输入输出的尺寸相同
        # self.reset_gate = nn.Conv2d(input_channel*2, hidden_channel, kernel_size=kernel_size, stride=1, padding=padding)
        # self.update_gate = nn.Conv2d(input_channel*2, hidden_channel, kernel_size=kernel_size, stride=1, padding=padding)
        # self.out_gate = nn.Conv2d(input_channel*2, hidden_channel, kernel_size=kernel_size, stride=1, padding=padding)

        self.reset_gate = nn.Sequential(
            nn.Conv2d(input_channel*2, hidden_channel, kernel_size=kernel_size, stride=1, padding=padding),
            nn.LayerNorm([hidden_channel, width, width])
        )
        self.update_gate = nn.Sequential(
            nn.Conv2d(input_channel * 2, hidden_channel, kernel_size=kernel_size, stride=1, padding=padding),
            nn.LayerNorm([hidden_channel, width, width])
        )
        self.out_gate = nn.Sequential(
            nn.Conv2d(input_channel * 2, hidden_channel, kernel_size=kernel_size, stride=1, padding=padding),
            nn.LayerNorm([hidden_channel, width, width])
        )
        self.cross=CrossAttention3D(16, 4)#############################

    def forward(self, x, h):

        combined = torch.cat([x, h], dim=1)  # concatenate along channel axis
        # print(combined.shape)

        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))
        out_inputs = torch.tanh(self.out_gate(torch.cat([x, reset * h], dim=1)))

        h_new = (1 - update) * h + update * out_inputs
        # print(h_new.shape)

        # h_new,_=self.cross(h_new,h_new,h_new)

        return h_new

class LSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(LSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c_t + i_t * g_t
        o_t = torch.sigmoid(o_x + o_h + c_new)
        h_new = o_t * torch.tanh(c_new)
        return h_new, c_new