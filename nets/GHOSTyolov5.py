import torch
import torch.nn as nn
from nets.ConvNext import ConvNeXt_Small, ConvNeXt_Tiny
from nets.CSPdarknet import C3, Conv, CSPDarknet,C3_CA,C3Ghost,C3_Faster,C3RFEM,C3_Res2Block,C3_DCN
from nets.Swin_transformer import Swin_transformer_Tiny

from nets.EfficientFormerV2 import efficientformerv2_s0
from nets.attention import cbam_block, eca_block, se_block, CA_Block,ResBlock_HAM,EMA
attention_block = [se_block,cbam_block, eca_block, CA_Block,ResBlock_HAM,EMA]

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, backbone='cspdarknet', pretrained=False, input_shape=[640, 640]):
        super(YoloBody, self).__init__()
        #如果需要对主干特征进行注意力机制就打开   pi==0:关闭，pi==1:se,pi==2:cbam,pi==3:eca,pi==4:ca,pi==5:ResBlock_HAM
        self.pi             = 6
        self.backbone_feature_att = True
        depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        
        self.backbone_name  = backbone
        if backbone == "cspdarknet":
        
            self.backbone   = CSPDarknet(base_channels, base_depth, phi, pretrained)

        else:
           
            self.backbone       = {
                'convnext_tiny'         : ConvNeXt_Tiny,
                'convnext_small'        : ConvNeXt_Small,
                'swin_transfomer_tiny'  : Swin_transformer_Tiny,
                "efficientformerv2_s0"  : efficientformerv2_s0,
            }[backbone](pretrained=pretrained, input_shape=input_shape)
            in_channels         = {
                'convnext_tiny'         : [192, 384, 768],
                'convnext_small'        : [192, 384, 768],
                'swin_transfomer_tiny'  : [192, 384, 768],
                'efficientformerv2_s0'  : [48, 96, 176],
            }[backbone]
            feat1_c, feat2_c, feat3_c = in_channels 
            self.conv_1x1_feat1 = Conv(feat1_c, base_channels * 4, 1, 1)
            self.conv_1x1_feat2 = Conv(feat2_c, base_channels * 8, 1, 1)
            self.conv_1x1_feat3 = Conv(feat3_c, base_channels * 16, 1, 1)

       
        if 1 <= self.pi and self.pi <= 6:
            if self.backbone_feature_att :
                self.feat1_att      = attention_block[self.pi - 1](128)
                self.feat2_att      = attention_block[self.pi - 1](256)
                self.feat3_att   = attention_block[self.pi - 1](512)
            else:
                self.p5_att = attention_block[self.pi - 1](512)
                self.p4_att = attention_block[self.pi - 1](256)
                self.p3_att = attention_block[self.pi - 1](256)

           
            
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        # 全部使用C3_CA模块
        self.conv3_for_upsample1    =C3Ghost(base_channels * 16, base_channels * 8, base_depth, shortcut=False)
        # self.conv3_for_upsample1    = C3Ghost(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2    =C3Ghost(base_channels * 8, base_channels * 4, base_depth, shortcut=False)
        # self.conv3_for_upsample2    = C3Ghost(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  =C3Ghost(base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        # self.conv3_for_downsample1  =C3Ghost(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2  = C3Ghost(base_channels * 16, base_channels * 16, base_depth, shortcut=False)
        # self.conv3_for_downsample2  = C3Ghost(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        # 80, 80, 256 => 80, 80, 3 * (5 + num_classes) => 80, 80, 3 * (4 + 1 + num_classes)
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * (5 + num_classes) => 40, 40, 3 * (4 + 1 + num_classes)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes) => 20, 20, 3 * (4 + 1 + num_classes)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

    def forward(self, x):
        #  backbone
        #   80,80,256
        #   40,40,512
        #   20,20,1024
        feat1, feat2, feat3 = self.backbone(x)
        if self.backbone_name != "cspdarknet":
            feat1 = self.conv_1x1_feat1(feat1)
            feat2 = self.conv_1x1_feat2(feat2)
            feat3 = self.conv_1x1_feat3(feat3)

       
        if 1 <= self.pi and self.pi <= 6:
            if self.backbone_feature_att :
                feat1 = self.feat1_att(feat1)
                feat2 = self.feat2_att(feat2)
                feat3 = self.feat3_att(feat3)
            

        # 20, 20, 1024 -> 20, 20, 512
        P5          = self.conv_for_feat3(feat3)
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)
        # 40, 40, 512 -> 40, 40, 1024
        P4          = torch.cat([P5_upsample, feat2], 1)
        # 40, 40, 1024 -> 40, 40, 512
        P4          = self.conv3_for_upsample1(P4)
        # 40, 40, 512 -> 40, 40, 256
        P4          = self.conv_for_feat2(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 注意力机制
        #80, 80, 256-> 80, 80, 256
        # if self.backbone_feature_att == False:
        #     P3          =self.p3_att(P3)
        # 80, 80, 512 -> 80, 80, 256
        P3          = self.conv3_for_upsample2(P3)
        
        # 80, 80, 256 -> 40, 40, 256
        P3_downsample = self.down_sample1(P3)
        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)
        # 注意力机制
        #40, 40, 512-> 40, 40, 512
        # if self.backbone_feature_att == False:
        #     P4          =self.p4_att(P4)
        # 40, 40, 512 -> 40, 40, 512
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 512 -> 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5 = torch.cat([P4_downsample, P5], 1)
        # if self.backbone_feature_att == False:
        # # 20, 20, 1024 -> 20, 20, 1024
        #     P5          =self.p5_att(P5)

        # 20, 20, 1024 -> 20, 20, 1024
        P5 = self.conv3_for_downsample2(P5)

        #---------------------------------------------------#
     
        #   y3=(batch_size,75,80,80)
        #---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        #---------------------------------------------------#
   
        #   y2=(batch_size,75,40,40)
        #---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        #---------------------------------------------------#

        #   y1=(batch_size,75,20,20)
        #---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)
        return out0, out1, out2
    

