import torch
import torch.nn as nn
import math 
from nets.attention import EMA
# from timm.models.layers import DropPath
import numpy as np 


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        # 320, 320, 12 => 320, 320, 64
        return self.conv(
            # 640, 640, 3 => 320, 320, 12
            torch.cat(
                [
                    x[..., ::2, ::2], 
                    x[..., 1::2, ::2], 
                    x[..., ::2, 1::2], 
                    x[..., 1::2, 1::2]
                ], 1
            )
        )

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat(
            (
                self.m(self.cv1(x)), 
                self.cv2(x)
            )
            , dim=1))
    
# CA 注意力机制
class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()
        
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.relu   = nn.ReLU()
        self.bn     = nn.BatchNorm2d(channel//reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
        _, _, h, w = x.size()
        
        x_h = torch.mean(x, dim = 3, keepdim = True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim = 2, keepdim = True)
 
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
 
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
 
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
 
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
    
# C3_CA 模块
class C3_CA(nn.Module):
     # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3_CA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.att=CA_Block(c2)
    def forward(self, x):

        x= self.cv3(torch.cat(
            (
                self.m(self.cv1(x)), 
                self.cv2(x)
            )
            , dim=1))
        return self.att(x)
    
 #  C3Ghost模块轻量化模型---------------------------------------------------------------------------------------------------------

 # 深度可分离卷积   
class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)
 
    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)
    
    
class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
 
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[GhostBottleneck(c_, c_) for _ in range(n)])
# ---------------------------------------------------------------------------------------------------------

# faster-c3--------------------------------------------------------------------------------------
class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

class Faster_Block(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )
        
        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

class C3_Faster(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Faster_Block(c_, c_) for _ in range(n)))
# --------------------------------------------------------------------------------------------------------------------------------------------
# C3RFEM--------------------------------------------------------------------------------------------------------------------------------------
class TridentBlock(nn.Module):
    def __init__(self, c1, c2, stride=1, c=False, e=0.5, padding=[1, 2, 3], dilate=[1, 2, 3], bias=False):
        super(TridentBlock, self).__init__()
        self.stride = stride
        self.c = c
        c_ = int(c2 * e)
        self.padding = padding
        self.dilate = dilate
        self.share_weightconv1 = nn.Parameter(torch.Tensor(c_, c1, 1, 1))
        self.share_weightconv2 = nn.Parameter(torch.Tensor(c2, c_, 3, 3))

        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c2)

        self.act = nn.SiLU()

        nn.init.kaiming_uniform_(self.share_weightconv1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.share_weightconv2, nonlinearity="relu")

        if bias:
            self.bias = nn.Parameter(torch.Tensor(c2))
        else:
            self.bias = None

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward_for_small(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride, padding=self.padding[0],
                                   dilation=self.dilate[0])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward_for_middle(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride, padding=self.padding[1],
                                   dilation=self.dilate[1])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward_for_big(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride, padding=self.padding[2],
                                   dilation=self.dilate[2])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward(self, x):
        xm = x
        base_feat = []
        if self.c is not False:
            x1 = self.forward_for_small(x)
            x2 = self.forward_for_middle(x)
            x3 = self.forward_for_big(x)
        else:
            x1 = self.forward_for_small(xm[0])
            x2 = self.forward_for_middle(xm[1])
            x3 = self.forward_for_big(xm[2])

        base_feat.append(x1)
        base_feat.append(x2)
        base_feat.append(x3)

        return base_feat

class RFEM(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, stride=1):
        super(RFEM, self).__init__()
        c = True
        layers = []
        layers.append(TridentBlock(c1, c2, stride=stride, c=c, e=e))
        c1 = c2
        for i in range(1, n):
            layers.append(TridentBlock(c1, c2))
        self.layer = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.layer(x)
        out = out[0] + out[1] + out[2] + x
        out = self.act(self.bn(out))
        return out

class C3RFEM(C3):
    # C3 module with RFEM
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(RFEM(c_, c_, n=1, e=e) for _ in range(n)))
# --------------------------------------------------------------------------------------------------------------------------------------------
# C3-res2block-----------------------------------------------------------------------------------------------------------------------------------------
class res2block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, shortcut, baseWidth=26, scale = 4):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(res2block, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = Conv(inplanes, width*scale, k=1)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        convs = []
        for i in range(self.nums):
          convs.append(Conv(width, width, k=3))
        self.convs = nn.ModuleList(convs)

        self.conv3 = Conv(width*scale, planes * self.expansion, k=1, act=False)

        self.silu = nn.SiLU(inplace=True)
        self.scale = scale
        self.width  = width
        self.shortcut = shortcut

    def forward(self, x):
        if self.shortcut:
            residual = x
        out = self.conv1(x)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1:
          out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        if self.shortcut:
            out += residual
        out = self.silu(out)
        return out

class C3_Res2Block(C3):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(res2block(c_, c_, shortcut) for _ in range(n)))
#-------------------------------------------------------------------------------------------------------------------------
#C3-DCN-------------------------------------------------------------------------------------------------------------------------
class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

class Bottleneck_DCN(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DCNv2(c_, c2, 3, 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3_DCN(C3):
    # C3 module with DCNv2
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DCN(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
#----------------------------------------------------------------------------------------------------------------------

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
        
class CSPDarknet(nn.Module):
    def __init__(self, base_channels, base_depth, phi, pretrained):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道base_channels是64
        #-----------------------------------------------#

        #-----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # #-----------------------------------------------#
        # self.att_EMA=EMA(512)
        self.stem       = Focus(3, base_channels, k=3)
        
        #-----------------------------------------------#
        #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        #-----------------------------------------------#
        self.dark2 = nn.Sequential(
            # 320, 320, 64 -> 160, 160, 128
            Conv(base_channels, base_channels * 2, 3, 2),
            # 160, 160, 128 -> 160, 160, 128
            C3Ghost(base_channels * 2, base_channels * 2, base_depth),
        )
        
        #-----------------------------------------------#
        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        #                   在这里引出有效特征层80, 80, 256
        #                   进行加强特征提取网络FPN的构建
        #-----------------------------------------------#
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C3Ghost(base_channels * 4, base_channels * 4, base_depth * 3),
        )

        #-----------------------------------------------#
        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        #                   在这里引出有效特征层40, 40, 512
        #                   进行加强特征提取网络FPN的构建
        #-----------------------------------------------#
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C3Ghost(base_channels * 8, base_channels * 8, base_depth * 3),
        )
        
        #-----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        #-----------------------------------------------#
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            SPP(base_channels * 16, base_channels * 16),
            C3Ghost(base_channels * 16, base_channels * 16, base_depth, shortcut=False),

        )
        if pretrained:
            url = {
                's' : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_s_backbone.pth',
                'm' : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_m_backbone.pth',
                'l' : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_l_backbone.pth',
                'x' : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_x_backbone.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from ", url.split('/')[-1])
            
    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3的输出为80, 80, 256，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为40, 40, 512，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3
    
    # if __name__ =="__main__":
    #         inputs = torch.randn(( 1,128, 160, 160))
    #         model = C3_CA( 128, 128, 1)
    #         res=model(inputs)
    #         print( res.shape)
            
            # for i in res:
            #     print(i.size())
        
