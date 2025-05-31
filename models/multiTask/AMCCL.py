import math
import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.BertTextEncoder import BertTextEncoder

__all__ = ['AMCCL']

from models.subNets.FeatureNets import SubNet


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = torch.mean(x, dim=2).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class AMC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AMC, self).__init__()
        base_channels = out_channels // 3
        remainder = out_channels % 3
        self.conv1_out = base_channels + (1 if remainder > 0 else 0)
        self.conv3_out = base_channels + (1 if remainder > 1 else 0)
        self.conv5_out = base_channels
        self.kernel_size_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels, 3)
        )
        self.conv1 = nn.Conv1d(in_channels, self.conv1_out, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, self.conv3_out, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, self.conv5_out, kernel_size=5, padding=2)
        self.se_block = SEBlock(self.conv1_out + self.conv3_out + self.conv5_out)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        kernel_weights = F.softmax(self.kernel_size_predictor(x), dim=1)
        c1 = self.conv1(x) * kernel_weights[:, 0].unsqueeze(1).unsqueeze(2)
        c3 = self.conv3(x) * kernel_weights[:, 1].unsqueeze(1).unsqueeze(2)
        c5 = self.conv5(x) * kernel_weights[:, 2].unsqueeze(1).unsqueeze(2)
        fused = torch.cat([c1, c3, c5], dim=1)
        fused = self.se_block(fused)
        fused = self.adaptive_pool(fused).squeeze(2)
        return fused


class AMCCL(nn.Module):
    def __init__(self, args):
        super(AMCCL, self).__init__()
        self.aligned = args.need_data_aligned
        self.relu = nn.ReLU()

        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        audio_in, video_in = args.feature_dims[1:]
        audio_len, video_len = args.seq_lens[1:]
        self.audio_model = AuVi_Encoder(audio_in, args.a_encoder_heads, args.a_encoder_layers, audio_len, args.device)
        self.video_model = AuVi_Encoder(video_in, args.v_encoder_heads, args.v_encoder_layers, video_len, args.device)

        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(args.post_text_dim + args.post_audio_dim + args.post_video_dim,
                                             args.post_fusion_dim)
        self.AMC = AMC(in_channels=args.post_text_dim, out_channels=args.post_text_dim)
        skip_connection_length = args.post_text_dim + args.post_audio_dim + args.post_video_dim + args.post_text_dim
        self.skip_connection_BatchNorm = nn.BatchNorm1d(skip_connection_length)
        self.skip_connection_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_skip_connection = nn.Linear(skip_connection_length, args.post_fusion_dim * 3)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim * 3, args.post_fusion_dim * 2)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim * 2, 1)
        self.fusionBatchNorm = nn.BatchNorm1d(args.post_fusion_dim * 2)

        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.text_skip_net = unimodal_skip_net(input_channel=args.text_out, enlager=args.post_text_dim,
                                               reduction=args.skip_net_reduction)
        self.post_text_layer_1 = nn.Linear(args.text_out, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)

        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.audio_skip_net = unimodal_skip_net(input_channel=args.audio_out, enlager=args.post_text_dim,
                                                reduction=args.skip_net_reduction)
        self.post_audio_layer_1 = nn.Linear(args.audio_out, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)

        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.video_skip_net = unimodal_skip_net(input_channel=args.video_out, enlager=args.post_text_dim,
                                                reduction=args.skip_net_reduction)
        self.post_video_layer_1 = nn.Linear(args.video_out, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)

    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video

        text = self.text_model(text)
        audio = self.audio_model(audio)
        video = self.video_model(video)

        text_h = self.post_text_dropout(text[:, 0, :])
        text_skip = self.text_skip_net(text)
        text_h = self.relu(self.post_text_layer_1(text_h))

        audio_h = self.post_audio_dropout(audio[:, -1, :])
        audio_skip = self.audio_skip_net(audio)
        audio_h = self.relu(self.post_audio_layer_1(audio_h))

        video_h = self.post_video_dropout(video[:, -1, :])
        video_skip = self.video_skip_net(video)
        video_h = self.relu(self.post_video_layer_1(video_h))

        fusion_h = torch.cat([text_h.unsqueeze(-1), audio_h.unsqueeze(-1), video_h.unsqueeze(-1)], dim=-1)
        fusion_h = self.AMC(fusion_h)

        fusion_h = torch.cat([fusion_h, text_skip, audio_skip, video_skip], dim=-1)
        fusion_h = self.skip_connection_dropout(fusion_h)
        fusion_h = self.post_fusion_layer_skip_connection(fusion_h)

        x_f = self.relu(self.post_fusion_layer_2(fusion_h))
        x_f = self.fusionBatchNorm(x_f)
        output_fusion = self.post_fusion_layer_3(x_f)

        res = {
            'M': output_fusion,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
        }
        return res


class AuVi_Encoder(nn.Module):
    def __init__(self, hidden_size, nhead=1, num_layers=1, max_length=1, device=None):
        super(AuVi_Encoder, self).__init__()
        self.position_embbeding = PositionalEncoding(hidden_size, 0.1, device, max_length)
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, x):
        x = self.position_embbeding(x)
        output = self.transformer_encoder(x)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x

class unimodal_skip_net(nn.Module):
    def __init__(self, input_channel, enlager, reduction=8):
        super(unimodal_skip_net, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_channel, input_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(input_channel // reduction, enlager, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = x.transpose(2, 1)
        output = output.unsqueeze(2)
        output = self.avg_pool(output).squeeze()
        output = self.fc(output)
        output = output.squeeze()
        return output