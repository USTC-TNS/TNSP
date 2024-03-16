# -*- coding: utf-8 -*-
# This file is copied from:
# https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
# MIT License
# Copyright (c) 2020 Peter Tatkowski
# Copyright (c) 2024 Hao Zhang<zh970205@mail.ustc.edu.cn>

import numpy as np
import torch
import torch.nn as nn


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding3D(nn.Module):

    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        inv_freq = 1.0 / (10000**(torch.arange(0, self.channels, 2, device=tensor.device).float() / self.channels))
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=inv_freq.dtype)
        pos_z = torch.arange(z, device=tensor.device, dtype=inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, :, :self.channels] = emb_x
        emb[:, :, :, self.channels:2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
