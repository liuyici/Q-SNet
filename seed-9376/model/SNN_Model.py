import torch
from einops import rearrange
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn

from model.snn_layers import first_order_low_pass_layer, neuron_layer

class SLR_layer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(SLR_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        r = input.norm(dim=1).detach()[0]
        cosine = F.linear(input, F.normalize(self.weight), r * torch.tanh(self.bias))
        output = cosine
        return output

class SNN_Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.batchsize = cfg.batchsize
        self.sub = cfg.d_classes

        self.axon1 = first_order_low_pass_layer((cfg.dim_1,), cfg.length, self.batchsize, cfg.tau_m,
                                                cfg.train_coefficients)
        self.snn1 = neuron_layer(cfg.dim_1, cfg.dim_2, cfg.length, self.batchsize, cfg.tau_m, cfg.train_bias,
                                 cfg.membrane_filter)

        self.axon2 = first_order_low_pass_layer((cfg.dim_2,), cfg.length, self.batchsize, cfg.tau_m,
                                                cfg.train_coefficients)
        self.snn2 = neuron_layer(cfg.dim_2, 128, cfg.length, self.batchsize, cfg.tau_m, cfg.train_bias,
                                 cfg.membrane_filter)

        self.axon3 = first_order_low_pass_layer((128,), cfg.length, self.batchsize, cfg.tau_m, cfg.train_coefficients)
        self.snn3 = neuron_layer(128, 100, cfg.length, self.batchsize, cfg.tau_m, cfg.train_bias, cfg.membrane_filter)

        self.dropout1 = torch.nn.Dropout(p=0.1, inplace=False)
        self.dropout2 = torch.nn.Dropout(p=0.1, inplace=False)
        self.linear = torch.nn.Linear(128 * cfg.window_size, cfg.n_classes)
        self.feat = nn.Sequential(
            nn.Linear(128 * cfg.window_size, 128),
            nn.ELU(),
            nn.Dropout(0.3),
        )
        self.fc2 = SLR_layer(128, cfg.n_classes)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """
        # 输入: [batch, 12, 310] → [batch, 310, 12]
        inputs = rearrange(inputs, 'b c h -> b h c')

        # 初始化状态
        axon1_states = self.axon1.create_init_states()
        snn1_states = self.snn1.create_init_states()

        # 前向传播
        axon1_out, axon1_states = self.axon1(inputs, axon1_states)   # [B, 310, 12] → [B, 310]
        spike_l1, snn1_states = self.snn1(axon1_out, snn1_states)    # [B, 128, 12]
        drop_1 = self.dropout1(spike_l1)                              # dropout

        # 展平 + 分类
        spike_fea = drop_1.reshape(drop_1.size(0), -1)               # [B, 128*window_size]
        feat = self.feat(spike_fea)                                     # [B, 128]
        out = self.fc2(feat)
        #out = self.linear(spike_fea)                                 # [B, n_classes]

        return out, feat
    
