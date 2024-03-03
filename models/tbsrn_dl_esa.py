import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np
import warnings
import math, copy
warnings.filterwarnings("ignore")

from models.tps_spatial_transformer import TPSSpatialTransformer
from models.stn_head import STNHead


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # print(features)
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)  # [64,16,64]
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)  # [16,1]
    pos_h = torch.arange(0., height).unsqueeze(1)  # [64,1]
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe  # [64,16,64]


class FeatureEnhancer(nn.Module):

    def __init__(self):
        super(FeatureEnhancer, self).__init__()
        self.multihead = MultiHeadedAttention(h=4, d_model=128, dropout=0.1)
        self.gru_h = GruBlock_h(64)
        self.gru_h2 = GruBlock_h(128)
        self.gru_v = GruBlock_v(64)
        self.gru_v2 = GruBlock_v(128)
        self.mul_layernorm1 = LayerNorm(features=128)

        self.pff = PositionwiseFeedForward(128, 128)
        self.mul_layernorm3 = LayerNorm(features=128)

        self.linear = nn.Linear(128, 64)

        # --------------------------------------------#
        #   添加注意力机制
        self.block_attention = CALayer(
            128)  # CALayer(128)#CAB(128, 3, 16, bias=False, act=nn.PReLU())#attention_blocks[3](128)
        self.block = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # self.ESA = ESA(128)
        # --------------------------------------------#

    def forward(self, conv_feature):
        '''
        text : (batch, seq_len, embedding_size)
        global_info: (batch, embedding_size, 1, 1)
        conv_feature: (batch, channel, H, W)
        '''
        batch = conv_feature.shape[0]  # [2,64,1024]
        # position2d = positionalencoding2d(64, 16, 64).float().cuda()
        # position2d = position2d.repeat(batch, 1, 1, 1)  # #[2,64,1024]
        # Concatenate with 2-D PE
        # conv_feature = torch.cat([conv_feature, position2d], 1)  # batch, 128(64+64), 16, 64
        Feature_v = self.gru_v(conv_feature)  # b , h, w, c # BCHW -> BHWC
        Feature_h = self.gru_h(conv_feature)  # b , h ,w ,c
        size = conv_feature.shape
        result = torch.cat([Feature_h, Feature_v], 1)
        result = result.view(size[0], 2 * size[1], -1)
        result = result.permute(0, 2, 1).contiguous()
        origin_result = result

        result = self.mul_layernorm1(origin_result + self.multihead.forward(result, result, result, mask=None)[0])
        origin_result = result  # [2,1024,128]
        # Postion-Wise Feed-Forward
        result = self.mul_layernorm3(origin_result + self.pff(result))
        result = self.linear(result)  # [2,1024,64]
        # Reshape
        result = result.permute(0, 2, 1).contiguous()  # [2,64,1024]
        result = result.resize(size[0], size[1], size[2], size[3])

        return result  # result.permute(0, 2, 1).contiguous()


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, align=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, align=align)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x), attention_map


def attention(query, key, value, mask=None, dropout=None, align=None):
    "Compute 'Scaled Dot Product Attention'"

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        # print(mask)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    else:
        pass

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TBSRN(nn.Module):
    def __init__(self, scale_factor=2, width=128, height=32, STN=True, srb_nums=5, mask=False, hidden_units=32,
                 input_channel=3, phi=0):
        super(TBSRN, self).__init__()

        self.conv = nn.Conv2d(input_channel, 3, 3, 1, 1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.mba = MBA(2 * hidden_units)
        #self.dff = DFF(2 * hidden_units)
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
            # nn.ReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(2 * hidden_units))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(

                    nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2 * hidden_units)
                ))
        # self.dff = nn.Conv2d(2 * hidden_units * 5, 2 * hidden_units, kernel_size=1, padding=0)

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')
        # --------------------------------------------#
        #   添加注意力机制
        self.phi = phi
        #if phi >= 1 and phi <= 4:
        #    self.block1_attention = attention_blocks[phi - 1](64)
        # --------------------------------------------#

    def forward(self, x):
        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}
        # --------------------------------------------#
        #   添加注意力机制
        if self.phi >= 1 and self.phi <= 4:
            block['1'] = self.block1_attention(block['1'])
        # --------------------------------------------#

        # for i in range(self.srb_nums + 1):
        #     block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        for i in range(self.srb_nums):
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])
        # block[]存结果  self.block_i 取block
        # mba = self.mba(block['6'])
        #dff = self.dff(torch.cat([block['2'], block['3'], block['4'], block['5'], block['6']], 1))
        # block[str(self.srb_nums + 2)] = getattr(self, 'block%d' % (self.srb_nums + 2)) \
        #     (block['6'])
        block[str(self.srb_nums + 2)] = getattr(self, 'block%d' % (self.srb_nums + 2)) \
            (block['6'])
        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))
        output = torch.tanh(block[str(self.srb_nums + 3)])
        return output


class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        #self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        #self.gru2 = GruBlock(channels, channels)
        self.feature_enhancer = FeatureEnhancer()
        #self.CA = CALayer(128)  # CAB(128, 3, 16, bias=False, act=nn.PReLU())
        self.block = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.TFF = TFF(channels, channels)
        #self.esa1 = ESA(channels)
        #self.esa2 = ESA(channels)

    def forward(self, x):
        # x = self.esa1(x)
        # Conv2d(64, 64, kernel size=(3, 3), stride=(1, 1), padding=(1, 1))
        residual = self.conv1(x)
        # BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track running stats=True)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        # Conv2d(64, 64, kernel size=(3, 3), stride=(1, 1), padding=(1, 1))
        residual = self.conv2(residual)
        # BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track running stats=True)
        residual = self.bn2(residual)

        # size = residual.shape  #[2,64,16,64]
        # residual = residual.view(size[0],size[1],-1)  # [2,64,1024]  size[0] 和 size[1] 是先前定义的维度大小。最后一个维度 -1 表示根据张量的总元素数量自动推断该维度的大小。
        ############################################################
        ##                    特征增强
        ## Concatenate with 2-D PE  --->Flatten ---> Self-Attention ---> Postion-Wise Feed-Forward
        ##############################################################
        residual = self.feature_enhancer(residual)  # [2,64,1024]
        residual = x + residual
        residual = self.TFF(residual)
        # residual = self.esa2(residual)
        return residual


####纹理特征融合模块  10——3改
class TFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TFF, self).__init__()
        self.conv_begin = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        self.vec1_conv3x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=(1, 3), padding=(0, 1))
        self.vec2_conv3x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=(1, 3), padding=(0, 1))
        self.hor1_conv1x3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                      kernel_size=(3, 1), padding=(1, 0))
        self.hor2_conv1x3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                      kernel_size=(3, 1), padding=(1, 0))

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=3, padding=1)
        self.ca = CALayer(3 * in_channels)
        self.ca1 = CALayer(in_channels)

        self.global_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                     kernel_size=3, padding=1)
        self.conv_end = nn.Conv2d(in_channels=3 * in_channels, out_channels=out_channels,
                                  kernel_size=3, padding=1)
        #self.conv_last = nn.Conv2d(in_channels=3 * in_channels, out_channels=out_channels,kernel_size=1, padding=0)
        self.esa = ESA(in_channels)
        #self.esa1 = ESA(3*in_channels)
        #self.mba = MBA(in_channels)

    def forward(self, x):
        feature = self.conv_begin(x)

        feature_global = self.global_conv(feature)

        feature1 = self.vec1_conv3x1(feature)
        feature1 = self.vec2_conv3x1(feature1)

        feature2 = self.hor1_conv1x3(feature)
        feature2 = self.hor2_conv1x3(feature2)

        feature3 = self.conv1(feature)
        feature3 = self.conv2(feature3)

        feature_local = torch.cat([feature1, feature2, feature3], 1)
        # feature_local = self.ca(feature_local)
        #feature_local = self.esa1(feature_local)
        feature_local = self.conv_end(feature_local)

        #feature_local = self.esa(feature_local ) * self.ca1(feature_local )
        # feature = self.esa(feature_local + feature_global + x  )
        #feature_local = self.mba(feature_local+feature_global)
        feature_local = feature_local + feature_global
        feature_local = self.esa(feature_local)
        return feature_local + x

    ###密集残差连接


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=64, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 1, 1, 0, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 1, 1, 0, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 1, 1, 0, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 1, 1, 0, bias=bias)
        self.conv6 = nn.Conv2d(nf + 4 * gc, nf, 1, 1, 0, bias=bias)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.lrelu = nn.PReLU()
        self.srb1 = RecurrentResidualBlock(nf)
        self.srb2 = RecurrentResidualBlock(nf)
        self.srb3 = RecurrentResidualBlock(nf)
        self.srb4 = RecurrentResidualBlock(nf)
        self.srb5 = RecurrentResidualBlock(nf)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.srb1(x)

        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))  # 前面
        x2 = self.srb2(x2)

        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x3 = self.srb3(x3)

        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x4 = self.srb4(x4)

        x5 = self.lrelu(self.conv5(torch.cat((x, x1, x2, x3, x4), 1)))
        x5 = self.srb5(x5)

        x5 = self.conv6(torch.cat((x1, x2, x3, x4, x5), 1))
        return x5 + x


#  垂直
class GruBlock_v(nn.Module):
    def __init__(self, n_feat, out_channels=64):
        super(GruBlock_v, self).__init__()
        self.gru = nn.GRU(out_channels, 32, bidirectional=True, batch_first=True, )
        self.conv2 = nn.Conv2d(n_feat, out_channels, kernel_size=1, padding=0)

    def forward(self, x):  # b,c,h,w
        # x.transpose(-1, -2).contiguous() #  b,c,h,w 转置 宽高
        # x: b, c, h, w
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # b,h, w, c
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])  # b*h, w, c
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])  # b,h,w,c
        x = x.permute(0, 3, 1, 2).contiguous()  # b,c,h,w
        return x


#  水平
class GruBlock_h(nn.Module):
    def __init__(self, n_feat, out_channels=64):
        super(GruBlock_h, self).__init__()
        self.gru = nn.GRU(out_channels, 32, bidirectional=True, batch_first=True, )
        self.conv3 = nn.Conv2d(n_feat, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv3(x)
        x = x.transpose(-1, -2).contiguous()  # b, c ,w, h
        # x: b, c, w, h
        x = x.permute(0, 2, 3, 1).contiguous()  # b, w, h, c
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])  # b*w, h, c
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 2, 1).contiguous()
        return x


class DA(nn.Module):
    def __init__(self, in_channels):
        super(DA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0)
        self.prelu = nn.PReLU()
        self.CA = CALayer(in_channels)
        self.ESA = ESA(in_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv2(x)

        S = self.ESA(x)
        C = self.CA(x)

        D = torch.cat([S, C], 1)
        D = self.conv3(D)

        return x + D


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        # x: b, c, w, h
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # b, w, h, c
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])  # b*w, h, c
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class ESA(nn.Module):
    def __init__(self, n_feats):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = torch.nn.functional.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = torch.nn.functional.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


### 多注意力块
class MBA(nn.Module):
    def __init__(self, in_channels):
        super(MBA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.CA = CALayer(in_channels)
        self.ESA = ESA(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = self.conv1(x)

        E = self.ESA(residual)
        E = self.sigmoid(E)
        E = residual * E
        C = self.CA(residual)
        C = self.sigmoid(C)
        C = residual * C

        return C + E


class DFF(nn.Module):
    def __init__(self, in_channels):
        super(DFF, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels * 5, out_channels=in_channels,
                               kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


##########################################################################
if __name__ == '__main__':
    # net = NonLocalBlock2D(in_channels=32)
    # img = torch.zeros(7, 3, 16, 64)
    # embed()
    import torch

    # net = NonLocalBlock2D(in_channels=32)
    img = torch.zeros(7, 3, 16, 64)
    # embed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TBSRN(scale_factor=2, width=128, height=32,
                STN=True, mask=False, srb_nums=5, hidden_units=32).to(device)

    image = torch.rand(2, 3, 16, 64).to(device)
    print(net)
    print(net(image).shape)