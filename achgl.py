import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
from base_model import *
from base_model import SequenceModel
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module

import holoviews as hv

hv.extension('bokeh')
import numpy as np




# class GraphAttnMultiHead(Module):
class GraphAttnMultiHead(Module):
    def __init__(self, in_features, out_features, negative_slope=0.2, num_heads=4, bias=True, residual=True):
        super(GraphAttnMultiHead, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, num_heads * out_features))
        self.weight_u = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.weight_v = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.residual = residual
        if self.residual:
            self.project = nn.Linear(in_features, num_heads * out_features)
        else:
            self.project = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, num_heads * out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        # stdv = 1. / math.sqrt(self.weight_u.size(-1))
        self.weight_u.data.uniform_(-stdv, stdv)
        self.weight_v.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj_mat, requires_weight=True):
        support = torch.mm(inputs, self.weight)
        support = support.reshape(-1, self.num_heads, self.out_features).permute(dims=(1, 0, 2))
        f_1 = torch.matmul(support, self.weight_u).reshape(self.num_heads, 1, -1)
        f_2 = torch.matmul(support, self.weight_v).reshape(self.num_heads, -1, 1)
        logits = f_1 + f_2
        weight = self.leaky_relu(logits)
        if adj_mat.device != weight.device:
            adj_mat = adj_mat.to(weight.device)

        masked_weight = torch.mul(weight, adj_mat).to_sparse()
        attn_weights = torch.sparse.softmax(masked_weight, dim=2).to_dense()
        support = torch.matmul(attn_weights, support)
        support = support.permute(dims=(1, 0, 2)).reshape(-1, self.num_heads * self.out_features)
        if self.bias is not None:
            support = support + self.bias
        if self.residual:
            support = support + self.project(inputs)
        if requires_weight:
            return support, attn_weights
        else:
            return support, None


class HeterogeneousGATLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size, ):
        super(HeterogeneousGATLayer, self).__init__()
        self.w = nn.Sequential(nn.Linear(input_size, hidden_size),
                               nn.Linear(hidden_size, 1, bias=False))

    def forward(self,
                inputs,
                require_weight=False):
        attention = self.w(inputs)
        attention = torch.softmax(attention, dim=1)
        if require_weight:
            return torch.mul(attention, inputs).sum(dim=1), attention.squeeze()
        else:
            return torch.mul(attention, inputs).sum(dim=1)


class PairNorm(nn.Module):
    def __init__(self,
                 mode,
                 scale=1):
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x
        elif self.mode == 'PN':
            x = x - x.mean(dim=0)
            x = self.scale * x / (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        elif self.mode == 'PN-SI':
            x = x - x.mean(dim=0)
            x = self.scale * x / (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
        elif self.mode == 'PN-SCS':
            x = self.scale * x / (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt() - x.mean(dim=0)
        else:
            raise "No such a mode!"
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)
        att_output = []
        attn_matrix = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / self.temperature, dim=-1)
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)

            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output, attn_matrix


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        t_matrix = torch.softmax(torch.matmul(q, k.transpose(1, 2)), dim=-1)
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)

            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output, t_matrix


class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output / self.t, dim=-1)
        return self.d_output * output


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)  # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)

        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


class ACHGL(nn.Module):
    def __init__(self, d_feat=158, d_model=256, t_nhead=4, s_nhead=4, T_dropout_rate=0.5, S_dropout_rate=0.5,
                 gate_input_start_index=158, gate_input_end_index=221, cor=0.3, beta=None):
        super(ACHGL, self).__init__()
        # market
        self.encoding = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        self.d_feat = d_feat
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)  # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        # self.upstream_GAT = GATLayer(
        #     in_features=d_model,
        #     out_features=d_model,
        #     num_heads=4
        # )
        # self.downstream_GAT = GATLayer(
        #     in_features=d_model,
        #     out_features=d_model,
        #     num_heads=4
        # )
        self.device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        self.pos = PositionalEncoding(d_model=d_model)
        self.tempattn = TemporalAttention(d_model=d_model)
        self.L1 = nn.Linear(d_feat, d_model)
        self.L2 = nn.Linear(d_model, 1)
        self.l3 = nn.Linear(1024, 256)
        self.l5 = nn.Linear(224, 256)

        self.l4 = nn.Linear(d_model, 1)
        self.hetegat = HeterogeneousGATLayer(d_model, d_model)
        self.pair_norm = PairNorm(mode='PN-SCS')
        self.gat = GraphAttnMultiHead(d_model, d_model)
        self.tattn = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        self.sattn = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
        self.layers = nn.Sequential(
            # feature layer
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            TemporalAttention(d_model=d_model),
            )
        self.edge_importance = None

        self.shap_explainer_static = None
        self.shap_explainer_time = None


    def forward(self, x, y, require_weight=True,require_exp=True, require_moran=False):

        x = robust_zscore_norm_multidimensional(x, axis=(0, 1))
        x = torch.from_numpy(x).float()
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        src = x[:, :, :self.gate_input_start_index]  # N, T, D

        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]


        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

        src = self.L1(src)

        src = self.pos(src)
        src_pre = src
        # src_stream, _ = self.encoding(src)
        src, t_matrix = self.tattn(src)
        t_src = src

        # src_cor = src_cor[:, -1, :].squeeze()
        src_cor = src.reshape(x.size(0), -1)
        src_moran = src
        # cor_matrix_moran,cor_matrix_p,cor_matrix_z = calculate_moran(src_moran)
        # src_stream = src[:, -1, :].squeeze()
        # src_stream, _ = self.encoding(src)

        src_stream = src[:, -1, :].squeeze()

        cor_matrices = []
        D = 15
        for d in range(D):
            feature_series = src[:, :, d].detach().cpu().numpy()  # [N, T]
            cor_matrix_d = np.corrcoef(feature_series)  # [N, N]
            cor_matrices.append(cor_matrix_d)
        final_cor_matrix = np.mean(cor_matrices, axis=0)  #

        cor_matrix = torch.from_numpy(final_cor_matrix).float()

        # torch.save(cor_matrix_moran,'cor_matrix_moran.pt')
        # torch.save(cor_matrix_p, 'cor_matrix_p1.pt')
        # print('save successfully cor_matrix_p')
        # torch.save(cor_matrix_z, 'cor_matrix_z1.pt')

        pos_cor_matrix = (cor_matrix > 0.2).float()
        neg_cor_matrix = (cor_matrix < 0).float()


        # src = self.L1(src)
        # src_stream = src[:, -1, :].squeeze()
        src_s = src
        src, s_matrix = self.sattn(src)

        src = self.tempattn(src)
        pos_stream, pos_attn = self.gat(src_stream, pos_cor_matrix)
        neg_stream, neg_attn = self.gat(src_stream, neg_cor_matrix)
        # src_pos = pos_stream
        # src_neg = neg_stream

        # print(src.size())torch.Size([281, 8, 256])
        # src_upstream, attention_upstream = self.upstream_GAT(src[:, -1, :].squeeze(),
        #                                                    pos_adj,require_weight)
        # src_downstream, attention_downstream = self.downstream_GAT(src[:, -1, :].squeeze(),
        #                                                          neg_adj, require_weight)


        pos_stream = self.l3(pos_stream)
        # neg_stream = self.l3(neg_stream)
        # print(src.size(),src_downstream.size())torch.Size([281, 2048]) torch.Size([281, 1024])
        src = torch.stack((src, pos_stream,neg_stream), dim=1)

        # src, heterogeneous_attention = self.hetegat(src, require_weight)
        src,hete_attn = self.hetegat(src, require_weight)

        all_src = src
        src = self.pair_norm(src)
        # src = self.attn(src)
        # output = self.L2(src)
        output = self.l4(src)
        output = output.squeeze()

        return output

class ACHGLModel(SequenceModel):
    def __init__(
            self, d_feat: int = 20, d_model: int = 64, t_nhead: int = 8, s_nhead: int = 4, gate_input_start_index=None,
            gate_input_end_index=None, edge_index=None,
            T_dropout_rate=0.5, S_dropout_rate=0.5, beta=5.0, cor=0.3, **kwargs,
    ):
        super(ACHGLModel, self).__init__(**kwargs)
        self.shap_explainer = None  # 延迟初始化

        self.d_model = d_model
        self.d_feat = d_feat
        self.edge_index = edge_index
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta
        self.cor = cor
        self.init_model()


    def init_model(self):
        self.model = ACHGL(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                            T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                            gate_input_start_index=self.gate_input_start_index,
                            gate_input_end_index=self.gate_input_end_index, beta=self.beta, cor=self.cor)
        super(ACHGLModel, self).init_model()