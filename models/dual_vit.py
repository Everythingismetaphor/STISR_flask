import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np
from IPython import embed
import numpy as np

sys.path.append('../../')
sys.path.append('../../../')
from models.tps_spatial_transformer import TPSSpatialTransformer
from models.stn_head import STNHead
from models.my_transformer import InfoTransformer
from models.my_transformer import PositionalEncoding



SHUT_BN = False

class TSRNEncoder(nn.Module):
    def __init__(self, scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=True,
                 hidden_units=32
                 ):
        super(TSRNEncoder, self).__init__()
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

        # self.non_local = NonLocalBlock2D(64, 64)
        # block_ = [UpsampleBLock(2*hidden_units, 2) for _ in range(upsample_block_num)]
        # block_.append(nn.Conv2d(2*hidden_units, in_planes, kernel_size=9, padding=4))
        # setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
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

    def forward(self, x):
        # embed()
        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}

        for i in range(self.srb_nums + 1):
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        return [block['1'] + block[str(self.srb_nums + 2)]]


class TSRN(nn.Module):
    def __init__(self, scale_factor=2, width=128, height=32, STN=False, srb_nums=5, mask=True, hidden_units=32):
        super(TSRN, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2*hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
            # nn.ReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(2*hidden_units))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2*hidden_units, 2*hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2*hidden_units)
                ))

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2*hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2*hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height//scale_factor, width//scale_factor]
        tps_outputsize = [height//scale_factor, width//scale_factor]
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

    def forward(self, x):
        # embed()
        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}

        for i in range(self.srb_nums + 1):
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))
        output = torch.tanh(block[str(self.srb_nums + 3)])

        self.block = block

        # print("block_keys:", block.keys())
        # print("output:", output.shape)
        return output




class TPInterpreter(nn.Module):
    def __init__(
                self,
                t_emb,
                out_text_channels,
                output_size=(16, 64),
                feature_in=64,
                # d_model=512,
                t_encoder_num=1,
                t_decoder_num=2,
                 ):
        super(TPInterpreter, self).__init__()

        d_model = out_text_channels # * output_size[0] 64

        self.fc_in = nn.Linear(t_emb, d_model)
        self.fc_feature_in = nn.Linear(feature_in, d_model)

        self.activation = nn.PReLU()

        self.transformer = InfoTransformer(d_model=d_model,
                                          dropout=0.1,
                                          nhead=4,
                                          dim_feedforward=d_model,
                                          num_encoder_layers=t_encoder_num,
                                          num_decoder_layers=t_decoder_num,
                                          normalize_before=False,
                                          return_intermediate_dec=True, feat_height=output_size[0], feat_width=output_size[1])

        self.pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=5000)

        self.output_size = output_size
        self.seq_len = output_size[1] * output_size[0] #output_size[1] ** 2 #64*16
        self.init_factor = nn.Embedding(self.seq_len, d_model)

        self.masking = torch.ones(output_size)

        # self.tp_uper = InfoGen(t_emb, out_text_channels)

    def forward(self, image_feature, tp_input):

        # H, W = self.output_size
        x = tp_input   #b,37,1,26

        N_i, C_i, H_i, W_i = image_feature.shape
        H, W = H_i, W_i


        x_tar = image_feature

        # [1024, N, 64]
        x_im = x_tar.view(N_i, C_i, H_i * W_i).permute(2, 0, 1)

        device = x.device

        # x = x.permute(0, 3, 1, 2).squeeze(-1)  #b,26,37,1 -> b,26,37
        # print('x:', x.shape)
        # x = self.activation(self.fc_in(x))   #b,26,37  -> b,26,64
        # N, L, C = x.shape
        # x_pos = self.pe(torch.zeros((N, L, C)).to(device)).permute(1, 0, 2)
        # mask = torch.zeros((N, L)).to(device).bool()
        # x = x.permute(1, 0, 2)  # 26,b,64

        N_t, C_t, H_t, W_t = x.shape
        x = x.view(N_t, C_t, H_t * W_t).permute(2, 0, 1)  # N,C,L -> L,N,C
        x = self.activation(self.fc_in(x))
        L, N, C = x.shape
        x_pos = self.pe(torch.zeros((N, L, C)).to(device)).permute(1, 0, 2)
        mask = torch.zeros((N, L)).to(device).bool()

        # print("x:", x.shape)
        # print("x_im:", x_im.shape)
        text_prior = self.transformer(x, mask, self.init_factor.weight, x_pos, tgt=x_im, spatial_size=(H, W)) # self.init_factor.weight
        #text_prior = text_prior.mean(0)

        # [N, L, C] -> [N, C, H, W]
        text_prior = text_prior.permute(1, 2, 0).view(N, C, H, W)

        return text_prior


class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(64, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(64, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond

        # print("SFT:", x[0].shape, x[1].shape)

        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class InfoGenSFT(nn.Module):
    def __init__(
                self,
                t_emb,
                output_size
                 ):
        super(InfoGenSFT, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(t_emb, 512, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 128, 3, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, output_size, 3, (2, 1), padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(output_size)

        self.sft_layer = SFTLayer()

    def forward(self, t_embedding):

        # t_embedding += noise.to(t_embedding.device)

        x = F.relu(self.bn1(self.tconv1(t_embedding)))
        # print(x.shape)
        x = F.relu(self.bn2(self.tconv2(x)))
        # print(x.shape)
        x = F.relu(self.bn3(self.tconv3(x)))
        # print(x.shape)
        x = F.relu(self.bn4(self.tconv4(x)))
        # print(x.shape)

        return x, torch.zeros((x.shape[0], 1024, t_embedding.shape[-1])).to(x.device)



class InfoGen(nn.Module):
    def __init__(
                self,
                t_emb,
                output_size
                 ):
        super(InfoGen, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(t_emb, 512, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 128, 3, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, output_size, 3, (2, 1), padding=(1, 0), bias=False)
        self.bn4 = nn.BatchNorm2d(output_size)

    def forward(self, t_embedding):

        # t_embedding += noise.to(t_embedding.device)

        x = F.relu(self.bn1(self.tconv1(t_embedding)))
        # print(x.shape)
        x = F.relu(self.bn2(self.tconv2(x)))
        # print(x.shape)
        x = F.relu(self.bn3(self.tconv3(x)))
        # print(x.shape)
        x = F.relu(self.bn4(self.tconv4(x)))
        # print(x.shape)

        return x, torch.zeros((x.shape[0], 1024, t_embedding.shape[-1])).to(x.device)



class TSRN_TL(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=True,
                 hidden_units=32,
                 word_vec_d=300,
                 text_emb=3965, #37, #26+26+1 3965
                 out_text_channels=256): #256 32
        super(TSRN_TL, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlockTL(2 * hidden_units, out_text_channels)) #RecurrentResidualBlockTL

        self.feature_enhancer = None
        # From [1, 1] -> [16, 16]

        self.infoGen = InfoGen(text_emb, out_text_channels)

        if not SHUT_BN:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        nn.BatchNorm2d(2 * hidden_units)
                    ))
        else:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        # nn.BatchNorm2d(2 * hidden_units)
                    ))

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
                activation='none',
                input_size=self.tps_inputsize)

        self.block_range = [k for k in range(2, self.srb_nums+2)]

        # print("self.block_range:", self.block_range)

    def forward(self, x, text_emb=None, text_emb_gt=None):

        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)

        block = {'1': self.block1(x)}

        if text_emb is None:
            text_emb = torch.zeros(1, 3965, 1, 26).to(x.device) # 37 or 3965

        spatial_t_emb_gt, pr_weights_gt = None, None
        spatial_t_emb_, pr_weights = self.infoGen(text_emb)  # # ,block['1'], block['1'],

        spatial_t_emb = F.interpolate(spatial_t_emb_, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        # N, C, H, W = spatial_t_emb.shape

        # x_imfeat_pos = self.infoGen.pe(torch.zeros((N, H * W, C))
        #                                 .to(spatial_t_emb.device))\
        #                                 .reshape(N, H, W, C)\
        #                                 .permute(0, 3, 1, 2)

        # Reasoning block: [2, 3, 4, 5, 6]
        for i in range(self.srb_nums + 1):
            if i + 2 in self.block_range:
                # pred_word_vecs = self.w2v_proj(block[str(i + 1)])
                # all_pred_vecs.append(pred_word_vecs)
                # if not self.training:
                #     word_vecs = pred_word_vecs
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)], spatial_t_emb)
            else:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)])) #

        output = torch.tanh(block[str(self.srb_nums + 3)])

        self.block = block

        if self.training:
            ret_mid = {
                "pr_weights": pr_weights,
                "pr_weights_gt": pr_weights_gt,
                "spatial_t_emb": spatial_t_emb_,
                "spatial_t_emb_gt": spatial_t_emb_gt,
            }

            return output, ret_mid

        else:
            return output, pr_weights




class TSRN_TL_TRANS(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=True,
                 hidden_units=32,
                 word_vec_d=300,
                 text_emb=37, #37, #26+26+1 3965
                 out_text_channels=64, # 32 256
                 feature_rotate=False,
                 rotate_train=3.):
        super(TSRN_TL_TRANS, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlockTL(2 * hidden_units, out_text_channels)) #RecurrentResidualBlockTL

        self.infoGen = TPInterpreter(text_emb, out_text_channels, output_size=(height//scale_factor, width//scale_factor)) # InfoGen(text_emb, out_text_channels)
        self.trans = InfoGen(text_emb, out_text_channels)
        self.feature_rotate = feature_rotate
        self.rotate_train = rotate_train

        if not SHUT_BN:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        nn.BatchNorm2d(2 * hidden_units)
                    ))
        else:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        # nn.BatchNorm2d(2 * hidden_units)
                    ))

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

        self.block_range = [k for k in range(2, self.srb_nums + 2)]

    # print("self.block_range:", self.block_range)

    def forward(self, x, text_emb=None, text_emb_gt=None, feature_arcs=None, rand_offs=None):

        if self.stn and self.training:
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}

        if text_emb is None:
            text_emb = torch.zeros(1, 37, 1, 26).to(x.device) #37
        padding_feature = block['1']

        tp_map_gt, pr_weights_gt = None, None
        #text_emb,pa = self.trans(text_emb)
        text_emb = F.interpolate(text_emb, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        tp_map = self.infoGen(padding_feature, text_emb)
        # N, C, H, W

        # Reasoning block: [2, 3, 4, 5, 6]
        for i in range(self.srb_nums + 1):
            if i + 2 in self.block_range:
                # pred_word_vecs = self.w2v_proj(block[str(i + 1)])
                # all_pred_vecs.append(pred_word_vecs)
                # if not self.training:
                #     word_vecs = pred_word_vecs
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)], tp_map)
            else:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)])) #

        output = torch.tanh(block[str(self.srb_nums + 3)])

        self.block = block

        if self.training:
            ret_mid = {
                #"pr_weights": pr_weights,
                "pr_weights_gt": pr_weights_gt,
                "spatial_t_emb": tp_map,
                "spatial_t_emb_gt": tp_map_gt,
                "in_feat": block["1"],
                "trans_feat": tp_map
            }

            return output #, ret_mid

        else:
            return output

class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.gru1(residual.transpose(-1, -2)).transpose(-1, -2)
        # residual = self.non_local(residual)

        return self.gru2(x + residual)


class RecurrentResidualBlockTL(nn.Module):
    def __init__(self, channels, text_channels):
        super(RecurrentResidualBlockTL, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        # self.conv_proj = nn.Conv2d(channels + text_channels, channels, kernel_size=3, padding=1)

        self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        # self.gru2 = GruBlock(channels, channels)
        self.gru2 = GruBlock(channels, channels) # + text_channels

        # self.concat_conv = nn.Conv2d(channels + text_channels, channels, kernel_size=3, padding=1)
        self.mba = MBA(channels)
    def forward(self, x, text_emb):
        residual = self.conv1(x)
        if not SHUT_BN:
            residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        if not SHUT_BN:
            residual = self.bn2(residual)

        ############ Fusing with TL ############
        cat_feature = self.mba(residual,text_emb)
        # residual = self.concat_conv(cat_feature)
        ########################################
        # fused_feature = self.conv_proj(cat_feature)

        residual = self.gru1(cat_feature.transpose(-1, -2)).transpose(-1, -2)
        # residual = self.non_local(residual)

        return self.gru2(x + residual)


class RecurrentResidualBlockTLWEncod(nn.Module):
    def __init__(self, channels, text_channels):
        super(RecurrentResidualBlockTLWEncod, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels + text_channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)

        # self.concat_conv = nn.Conv2d(channels + text_channels, channels, kernel_size=3, padding=1)

    def forward(self, x, text_emb, imfeat_pos):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        ############ Fusing with TL ############
        cat_feature = torch.cat([residual, text_emb], 1)
        # residual = self.concat_conv(cat_feature)
        ########################################

        residual = self.gru1(cat_feature.transpose(-1, -2)).transpose(-1, -2)
        # residual = self.non_local(residual)

        return self.gru2(x + residual + imfeat_pos)


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

class ESALayer(nn.Module):
    def __init__(self, n_feats):
        super(ESALayer, self).__init__()
        f = n_feats // 16
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
        #self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.BN = nn.BatchNorm2d(in_channels)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, 1)
        )
        self.CA = CALayer(in_channels)
        self.ESA = ESALayer(in_channels)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x,e_tp):
        e_tp= self.BN(e_tp)
        cat_feature = torch.cat([x, e_tp], dim=1)
        feature = self.conv1x1(cat_feature)

        E = self.ESA(feature)

        C = self.CA(feature)
        output = E+C

        return output

# class CALayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(CALayer, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#                 nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#                 nn.Sigmoid()
#         )

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        # for i in range(2):
        modules_body.append(conv(n_feat, n_feat, (3, 1), padding=(1, 0), bias=bias))
        modules_body.append(nn.BatchNorm2d(n_feat))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, (1, 3), padding=(0, 1), bias=bias))
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        # print("shape:", res.shape, x.shape)
        # res = torch.nn.functional.interpolate(res, x.shape)
        res += x
        return res

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
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x


class ImFeat2WordVec(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImFeat2WordVec, self).__init__()
        self.vec_d = out_channels
        self.vec_proj = nn.Linear(in_channels, self.vec_d)

    def forward(self, x):

        b, c, h, w = x.size()
        result = x.view(b, c, h * w)
        result = torch.mean(result, 2)
        pred_vec = self.vec_proj(result)

        return pred_vec


if __name__ == '__main__':
    # net = NonLocalBlock2D(in_channels=32)
    img = torch.zeros(7, 3, 16, 64)
    embed()
