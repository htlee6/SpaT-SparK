import torch
import torch.nn as nn

from ..modules import ConvLSTMCell


class ConvLSTM_ModelSimple(nn.Module):
    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(ConvLSTM_ModelSimple, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(ConvLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size, configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, **kwargs):
        # PREPARATION

        device = frames_tensor.device
        """
        if not BTCHW:
            # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
            frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
            # mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        else:
            frames = frames_tensor.contiguous()
            # mask_true = mask_true.contiguous()
        """

        frames = frames_tensor.contiguous()

        batch = frames.shape[0]
        T = frames.shape[1]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(T):
            net = frames[:, t]
            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            # x_gen = self.conv_last(h_t[self.num_layers - 1])
            # next_frames.append(x_gen)

        for t in range(self.configs.aft_seq_length):
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            h_t[0], c_t[0] = self.cell_list[0](x_gen, h_t[0], c_t[0])
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            next_frames.append(x_gen)


        return torch.stack(next_frames, dim=1).contiguous()
