import argparse
import datetime
import os

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from torchmetrics.classification import MulticlassF1Score
import torch.nn.functional as F

class Multi_Scale_Spectral_Block(nn.Module):
    def __init__(self, dim, badaptive_filter):
        super().__init__()
        self.num_heads = 6
        self.dim_head = dim // self.num_heads

        # Learnable complex-valued spectral filters
        self.complex_weights = nn.Parameter(
            torch.randn(self.num_heads, self.dim_head, 2, dtype=torch.float32) * 0.02
        )
        self.complex_weights_high = nn.Parameter(
            torch.randn(self.num_heads, self.dim_head, 2, dtype=torch.float32) * 0.02
        )

        trunc_normal_(self.complex_weights_high, std=.02)
        trunc_normal_(self.complex_weights, std=.02)

        self.threshold_param = nn.Parameter(torch.ones(self.num_heads))

        self.badaptive_filter = badaptive_filter
        self.proj = nn.Linear(dim, dim)

    def create_adaptive_high_freq_mask(self, x_fft):
        N, B, _ = x_fft.shape
        H = self.num_heads
        D = N // H

        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        energy_reshaped = energy.view(H, D, B)

        median_energy = energy_reshaped.median(dim=1, keepdim=True)[0]

        epsilon = 1e-6
        normalized_energy = (energy_reshaped / (median_energy + epsilon)).view(N, B)

        thresh = self.threshold_param.view(H, 1, 1).repeat_interleave(D, dim=1).view(N, 1)

        # Generate an adaptive mask based on spectral energy
        adaptive_mask = ((normalized_energy > thresh).float() - thresh).detach() + thresh

        return adaptive_mask.unsqueeze(-1)


    def forward(self, x_in):
        B, N, C = x_in.shape
        H = self.num_heads
        D = self.dim_head

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Multi-head spectral decomposition
        x_split = x.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        x_flat = x_split.view(B * H, N, D)

        x_fft = torch.fft.rfft(x_flat, dim=1, norm='ortho')

        # Apply learnable spectral filtering
        weight = torch.view_as_complex(self.complex_weights)
        target_weight = weight.unsqueeze(0).repeat(B, 1, 1)
        weight_flat = target_weight.view(B * H, D)
        weight_final = weight_flat.unsqueeze(1).repeat(1, x_fft.shape[1], 1)

        x_weighted = x_fft * weight_final

        if self.badaptive_filter:

            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            # Adaptive enhancement for informative frequency components
            weight_high = torch.view_as_complex(self.complex_weights_high)
            target_weight_high = weight_high.unsqueeze(0).repeat(B, 1, 1)
            weight_high_flat = target_weight_high.view(B * H, D)
            weight_high_final = weight_high_flat.unsqueeze(1).repeat(1, x_fft.shape[1], 1)

            x_weighted2 = x_masked * weight_high_final
            x_weighted += x_weighted2


        # Transform back to the time domain
        x_irfft = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x_combined = x_irfft.view(B, H, N, D).permute(0, 2, 1, 3).contiguous()
        x_out = x_combined.view(B, N, C)

        x_out = self.proj(x_out)

        return x_out

class MSASB_layer(L.LightningModule):
    def __init__(self, MSASB, adaptive_filter, dim, drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.maasb = Multi_Scale_Spectral_Block(dim, adaptive_filter)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.MSASB = MSASB

    def forward(self, x):
        if self.MSASB:
            x = x + self.drop_path(self.maasb(self.norm1(x)))
        return x

class DAMR(L.LightningModule):
    def __init__(self, seq_len, hidden_dim, dropout_rate, depth, n_sensor,
                 out_channels, num_heads, MSASB, adaptive_filter, input_channels, conv_hidden):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, dropout_rate, depth)]  # stochastic depth decay rule

        self.msasb_blocks = nn.ModuleList([
            MSASB_layer(MSASB, adaptive_filter, dim=seq_len, drop=dropout_rate, drop_path=dpr[i])
            for i in range(depth)]
        )

        self.nonlin_map = Feature_extractor_1DCNN(input_channels, conv_hidden, out_channels)

        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=1, bias=False, dropout=dropout_rate)

        self.multiHead = MultiHeadProjector(conv_hidden * hidden_dim * n_sensor, 2*out_channels, num_heads)

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x, A):

        for msasb_blk in self.msasb_blocks:
            x = msasb_blk(x)

        x = x.unsqueeze(-1)
        B, N, T, D = x.shape
        x = x.reshape((B*N, T, D))
        x = self.nonlin_map(x)
        x = x.permute(0, 2, 1)
        h, _ = self.rnn(x)
        h = h.reshape((B, N, -1, 16))
        project = self.multiHead(h.flatten(start_dim=1))

        return project


class Feature_extractor_1DCNN(nn.Module):
     def __init__(self, input_channels, num_hidden, out_channels, kernel_size = 8, stride = 1, dropout = 0.15):
         super(Feature_extractor_1DCNN, self).__init__()

         self.conv_block1 = nn.Sequential(
             nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                       stride=stride, bias=False, padding=(kernel_size//2)),
             nn.BatchNorm1d(num_hidden),
             nn.ReLU(inplace=True),
             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
             nn.Dropout(dropout)
         )

         self.conv_block2 = nn.Sequential(
             nn.Conv1d(num_hidden, num_hidden*2, kernel_size=kernel_size, stride=1, bias=False, padding=2),
             nn.BatchNorm1d(num_hidden*2),
             nn.ReLU(inplace=True),
             nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
         )

         self.conv_block3 = nn.Sequential(
             nn.Conv1d(num_hidden*2, out_channels, kernel_size=kernel_size, stride=1, bias=False, padding=3),
             nn.BatchNorm1d(out_channels),
             nn.ReLU(inplace=True),
             nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
         )


     def forward(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        x = torch.transpose(x_in, -1,-2)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # print(x.size())
        return x


class MultiHeadProjector(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=1):
        super(MultiHeadProjector, self).__init__()
        self.num_heads = num_heads
        self.projection_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, input_dim // 16),
            nn.BatchNorm1d(input_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 16, output_dim)
        ) for _ in range(num_heads)])

    def forward(self, x):
        # Run the input through each projection head
        projections = [head(x) for head in self.projection_heads]
        return projections


class AdaptiveTemperature(nn.Module):
    def __init__(self, input_dim, num_heads=3, temp_init=0.1, lower_limit=1e-1, upper_limit=1.0):
        """
        Initialize the adaptive temperature module for multiple projection heads.
        :param input_dim: Dimensionality of the input.
        :param num_heads: Number of projection heads.
        :param temp_init: Initial temperature value.
        :param lower_limit: Lower limit of the temperature.
        :param upper_limit: Upper limit of the temperature.
        """
        super(AdaptiveTemperature, self).__init__()
        self.num_heads = num_heads
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        # Shared MLP layer to compute head-wise temperature (ϕ(·)) for each projection head
        self.shared_mlp_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim // 2, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, 1)
        ) for _ in range(num_heads)])

        self.shared_sim_matrix_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim // 2, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, 1)
        ) for _ in range(num_heads)])

    def forward(self, z_i_list, z_j_list):
        """
        Forward pass to compute adaptive temperatures for each projection head.
        :param z_i_list: List of feature vectors (from anchor time series) from each projection head.
        :param z_j_list: List of feature vectors (from positive or negative series) from each projection head.
        :return: List of scaled similarities.
        """
        scaled_similarities = []
        temps = []

        # Learn a separate similarity temperature for each head.
        for head_idx in range(self.num_heads):

            z_i = z_i_list[head_idx]
            z_j = z_j_list[head_idx]

            # Compute cosine similarity for the current head
            sim = torch.cosine_similarity(z_i, z_j)

            # Compute the dot product between z_i and z_j for temperature adjustment in current head
            dot_product = self.shared_mlp_heads[head_idx](torch.cat([z_i, z_j], dim=-1))

            dot_product_matrix = self.shared_sim_matrix_heads[head_idx](torch.cat([torch.cat([z_i, z_j], dim=0), torch.cat([z_i, z_j], dim=0)], dim=-1))

            # Apply sigmoid to get temperature between lower_limit and upper_limit for this head
            temp = torch.sigmoid(dot_product) * (self.upper_limit - self.lower_limit) + self.lower_limit
            temp_matrix = torch.sigmoid(dot_product_matrix) * (self.upper_limit - self.lower_limit) + self.lower_limit

            # Scale the similarity with the adaptive temperature
            scaled_sim = sim.unsqueeze(-1) / temp
            scaled_similarities.append(scaled_sim)
            temps.append(temp_matrix)

        return scaled_similarities, temps
