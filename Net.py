import torch
import torch.nn as nn
import torchvision.models as models
from backbone.pvt import pvt_v2_b2
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)

        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6, 12, 18], out_channels=32):
        super(ASPP, self).__init__()
        modules = []

        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1) 
        return self.project(res)


# Convolutional Block Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2



class MFF(nn.Module):
    def __init__(self, in_channel, dim,reduction_ratio=16,pixel_dim=32):
        super(MFF, self).__init__()
         
        self.ChannelGate = ChannelAttention(in_channel, reduction_ratio)
        self.SpatialGate = SpatialAttention()
        self.PixelGate = PixelAttention(pixel_dim)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
    def forward(self, img, depth):
        x = img + depth + (img * depth)
        cattn=self.ChannelGate(x)
        sattn=self.SpatialGate(x)
        pattn1=cattn+sattn
        pattn2=self.PixelGate(x,pattn1)
        result=x + pattn2 * img+(1 - pattn2) * depth
        result=self.conv(result)
        return result

class Enhanced_GGA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False):
        super(Enhanced_GGA, self).__init__()
        self.in_channels = in_channels
        
        # 1. 门控注意力生成（利用低级预测图）
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=1),  # 融合特征和低级预测图
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1),  # 输出注意力图
            nn.Sigmoid()
        )
        
        # 2. 多分支上下文探索（模拟 Focus 模块的 fp/fn）
        # 前景分支（fp）：强调目标区域
        self.fp_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, in_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        # 背景分支（fn）：抑制背景噪声
        self.fn_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=4, dilation=4),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, in_channels, 3, padding=5, dilation=5),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        # 3. 动态权重参数（类似 Focus 的 alpha/beta）
        self.alpha = nn.Parameter(torch.ones(1))  # 控制前景分支贡献
        self.beta = nn.Parameter(torch.ones(1))   # 控制背景分支贡献
        
        # 4. 输出卷积
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)

    def forward(self, in_feat, low_level_map):
        # Step 1: 生成注意力图（利用低级预测图）
        attention = self.gate_conv(torch.cat([in_feat, low_level_map], dim=1))
        # Step 2: 特征分割（前景和背景）
        f_feature = in_feat * attention   # 前景特征
        b_feature = in_feat * (1 - attention)  # 背景特征
        
        # Step 3: 多分支处理（模拟 Focus 的 fp/fn）
        fp_out = self.fp_branch(f_feature)
        fn_out = self.fn_branch(b_feature)
        
        # Step 4: 动态融合（类似 Focus 的 refine1/refine2）
        refined_feature = in_feat + self.alpha * fp_out - self.beta * fn_out
        
        # Step 5: 输出
        out_feat = self.out_conv(refined_feature)
        return out_feat


# Channel Attention Layer
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


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):  # act = ReLU or PReLU
        super(RCAB, self).__init__()
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


# Residual Feature Decoder
class RFD(nn.Module):
    def __init__(self, channel, kernel_size, reduction, bias, act, n_resblocks):
        super(RFD, self).__init__()
        modules_body = [RCAB(channel, kernel_size, reduction, bias=bias, act=act) for _ in range(n_resblocks)]
        modules_body.append(conv(channel, channel, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
class LaplacianEdge(nn.Module):
    def __init__(self):
        super(LaplacianEdge, self).__init__()
        # Laplacian 算子
        self.laplacian = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 初始化 Laplacian 算子的权重
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.laplacian.weight = nn.Parameter(laplacian_kernel, requires_grad=False)

    def forward(self, x):
        # 计算 Laplacian 边缘
        edge = self.laplacian(x)
        return edge

class MultiScaleEdgeFusion(nn.Module):
    def __init__(self, channels):
        super(MultiScaleEdgeFusion, self).__init__()
        # 边缘提取模块
        self.edge_extract = LaplacianEdge()
        
        # 多尺度特征融合模块
        self.conv1 = nn.Conv2d(channels[0] + 1, channels[0], kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels[1] + 1, channels[1], kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(channels[2] + 1, channels[2], kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(channels[3] + 1, channels[3], kernel_size=3, padding=1, bias=False)
        
        # 上采样和下采样
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

    def forward(self, x1, x2, x3, x4):
        # 提取边缘信息
        edge1 = self.edge_extract(x1.mean(dim=1, keepdim=True))
        edge2 = self.edge_extract(x2.mean(dim=1, keepdim=True))
        edge3 = self.edge_extract(x3.mean(dim=1, keepdim=True))
        edge4 = self.edge_extract(x4.mean(dim=1, keepdim=True))
        
        
        # 多尺度边缘信息融合
        x1 = torch.cat([x1, edge1], dim=1)
        x1 = self.conv1(x1)
        
        x2 = torch.cat([x2, edge2], dim=1)
        x2 = self.conv2(x2)
        
        x3 = torch.cat([x3, edge3], dim=1)
        x3 = self.conv3(x3)
        
        x4 = torch.cat([x4, edge4], dim=1)
        x4 = self.conv4(x4)
        
        # 将边缘信息融合到其他尺度的特征图中
        x2 = x2 + self.upsample(x1)
        x3 = x3 + self.upsample(x2)
        x4 = x4 + self.upsample(x3)
        
        return x1, x2, x3, x4

class UniNet(nn.Module):
    def __init__(self, channel=32, kernel_size=3, reduction=4, bias=False, act=nn.PReLU(), n_resblocks=2, iteration=3):
        super(UniNet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = r'/root/Camoflaged/UniNet/backbone/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.iteration = iteration

        self.DEConv_4 = ASPP(64)
        self.DEConv_3 = ASPP(128)
        self.DEConv_2 = ASPP(320)
        self.DEConv_1 = ASPP(512)

        self.edge_fusion = MultiScaleEdgeFusion(channels=[channel, channel, channel, channel])

        self.mff_4 = MFF(channel,channel)
        self.mff_3 = MFF(channel,channel)
        self.mff_2 = MFF(channel,channel)
        self.mff_1 = MFF(channel,channel)

        self.gate_1 = Enhanced_GGA(channel, channel)
        self.gate_2 = Enhanced_GGA(channel, channel)
        self.gate_3 = Enhanced_GGA(channel, channel)

        self.rfd_1 = RFD(channel, kernel_size, reduction, bias, act, n_resblocks)  # 32 x 22 x 22
        self.rfd_2 = RFD(2 * channel, kernel_size, reduction, bias, act, n_resblocks)  # 64 x 44 x 44
        self.rfd_3 = RFD(3 * channel, kernel_size, reduction, bias, act, n_resblocks)  # 96 x 88 x 88

        self.gate_conv = nn.Sequential(
            BasicConv2d(32, 1, 1),
            nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        )
        self.gate_conv_1 = BasicConv2d(32, 1, 1)
        self.gate_conv_2 = BasicConv2d(64, 1, 1)

        self.unsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.out = BasicConv2d(3 * channel, channel, 3, padding=1)
        self.pred = nn.Conv2d(channel, 1, 1)

        self.Fus = ASPP(channel)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_pred = nn.Conv2d(channel, 1, 1)
        self.final = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):  # 孪生网络，这里是batch拼接后的img和depth

        pvt = self.backbone(x)
        x4 = pvt[0]  # 64x176x176
        x3 = pvt[1]  # 128x88x88
        x2 = pvt[2]  # 320x44x44
        x1 = pvt[3]  # 512x22x22

        x4 = self.DEConv_4(x4)#torch.Size([4, 32, 176, 176])
        #print(f"x4:{x4.shape}")
        x3 = self.DEConv_3(x3)#x3:torch.Size([4, 32, 88, 88])
        #print(f"x3:{x3.shape}")
        x2 = self.DEConv_2(x2)#x2:torch.Size([4, 32, 44, 44])
        #print(f"x2:{x2.shape}")
        x1 = self.DEConv_1(x1)#x1:torch.Size([4, 32, 22, 22])
        #print(f"x1:{x1.shape}")

        x1, x2, x3, x4 = self.edge_fusion(x1, x2, x3, x4)

        x4_img, x4_depth = torch.chunk(x4, 2, dim=0)
        x3_img, x3_depth = torch.chunk(x3, 2, dim=0)
        x2_img, x2_depth = torch.chunk(x2, 2, dim=0)
        x1_img, x1_depth = torch.chunk(x1, 2, dim=0)

        stage_pred = list()
        coarse_pred = None  # 阶段预测结果
        for iter in range(self.iteration):
            x1 = self.mff_1(x1_img, x1_depth)
            #print(f"x1:{x1.shape}")#x1:torch.Size([2, 32, 22, 22])
            if coarse_pred == None:
                x1 = x1
            else:
                coarse_pred = self.gate_conv(coarse_pred)#coarse_pred:torch.Size([2, 1, 22, 22])
                #print(f"coarse_pred:{coarse_pred.shape}")
                x1 = self.gate_1(x1, coarse_pred)#x1:torch.Size([2, 32, 22, 22])
                #print(f"x1:{x1.shape}")
            x2_feed = self.rfd_1(x1)#x2_feed:torch.Size([2, 32, 22, 22])
            #print(f"x2_feed:{x2_feed.shape}")
            x2 = self.mff_2(x2_img, x2_depth)#x2:torch.Size([2, 32, 44, 44])
            #print(f"x2:{x2.shape}")
            if iter > 0:
                x2_gate = self.unsample_2(self.gate_conv_1(x2_feed))
                #print(f"x2:{x2_gate.shape}")
                x2 = self.gate_2(x2, x2_gate)
                #print(f"x2:{x2.shape}")
            x3_feed = self.rfd_2(torch.cat((x2, self.unsample_2(x2_feed)), dim=1))#x3:torch.Size([2, 64, 44, 44])
            #print(f"x3:{x3_feed .shape}")
            x3 = self.mff_3(x3_img, x3_depth)#x3:torch.Size([2, 32, 88, 88])
            #print(f"x3:{x3.shape}")
            if iter > 0:
                x3_gate = self.unsample_2(self.gate_conv_2(x3_feed))
                #print(f"x3:{x3_gate.shape}")
                x3 = self.gate_3(x3, x3_gate)
                #print(f"x3:{x3.shape}")
            x4_feed = self.rfd_3(torch.cat((x3, self.unsample_2(x3_feed)), dim=1))#x4:torch.Size([2, 96, 88, 88])
            #print(f"x4:{x4_feed.shape}")
            coarse_pred = self.out(x4_feed)#x4:torch.Size([4, 32, 176, 176])
            #print(f"x4:{x4.shape}")
            out_map = self.pred(coarse_pred)#outmap:torch.Size([2, 1, 88, 88])
            #print(f"outmap:{out_map.shape}")
            pred = F.interpolate(out_map, scale_factor=8, mode='bilinear')#pred:torch.Size([2, 1, 704, 704])
            #print(f"pred:{pred.shape}")
            stage_pred.append(pred)

        x4 = self.mff_4(x4_img, x4_depth)
        #print(f"x4:{x4.shape}")
        x4_out = self.downsample(x4)
        #print(f"x4:{x4.shape}")
        #x_in = torch.cat((coarse_pred, x4_out), dim=1)
        #print(f"xin:{x_in.shape}")
        refined_pred = self.Fus(x4_out)#ref:torch.Size([2, 32, 88, 88])
        #print(f"ref:{refined_pred.shape}")
        pred2 = self.out_pred(refined_pred)
        #print(f"pre2:{pred2.shape}")#pre2:torch.Size([2, 1, 88, 88])
        #final_pred = F.interpolate(pred2, scale_factor=8, mode='bilinear')
        final_pred=self.final(pred2)
        #print(f"fina:{final_pred.shape}")#fina:torch.Size([2, 1, 704, 704])
        return stage_pred, final_pred











