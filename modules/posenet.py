from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian
import pdb
from modules.util import ResBlock3d, SameBlock3d, UpBlock3d, DownBlock3d, ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
import math
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d

class PoseNet(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """
    def __init__(self, block_expansion, num_blocks, max_features, num_kp,scale_factor=1, kp_variance=0.01, reshape_depth=4, **args):
        super(PoseNet, self).__init__()

        downblocks = []
        input_dim = [num_kp,256,512,512,512]
        output_dim = [256,512,512,512,512]
        for i in range(len(input_dim)):
            downblocks.append(DownBlock3d(input_dim[i], output_dim[i]))
        self.downblocks = nn.ModuleList(downblocks)

        upblocks = []
        input_dim = [512,1024,1024,1024,512]
        output_dim = [512,512,512,256,256]
        for i in range(len(input_dim)):
            upblocks.append(UpBlock3d(input_dim[i], output_dim[i]))
        self.upblocks = nn.ModuleList(upblocks)
        
        self.sameblock = SameBlock3d(256+num_kp,256)
        self.kp_variance = kp_variance
        self.reshape_depth = reshape_depth
        self.scale_factor = scale_factor
        
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(3, self.scale_factor)
        
        self.second = SameBlock2d(self.reshape_depth*256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.third = SameBlock2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        
    def create_heatmap_representations(self, spatial_size, kp):
        gaussian_driving = kp2gaussian(kp, spatial_size=spatial_size, kp_variance=self.kp_variance)
        return gaussian_driving
    #只是用到了image的shape
    def forward(self, keypoint, image, **kwargs):
        # down encoder
        if self.scale_factor != 1:
            image = self.down(image) # 2d feature self.reshape_depth
        bs, c, h, w = image.shape    #(bz,512,w,h)
        spatial_size = (self.reshape_depth, h, w)
        out = self.create_heatmap_representations(spatial_size, keypoint)
        
        strength = keypoint['strength']
        out = strength * out
        encoder_map = [out] # wrap
        
        for i in range(len(self.downblocks)):
            out = self.downblocks[i](out)
            encoder_map.append(out)
        out = encoder_map.pop()
        outs = []
        for i in range(len(self.upblocks)):
            out = self.upblocks[i](out)
            skip = encoder_map.pop()
            out = torch.cat([out, skip], dim=1)
        out = self.sameblock(out)
        bs, c, d, h, w = out.shape
        out = out.contiguous().view(bs, c*d, h, w)
        out =  self.second(out)
        out = self.third(out)
        return out
        
class KeyPoseNet(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """
    def __init__(self, block_expansion, num_blocks, max_features, num_kp, scale_factor=1, kp_variance=0.01, reshape_depth=4, **args):
    
        super(KeyPoseNet, self).__init__()

        downblocks = []
        input_dim = [num_kp+3,256,512,512,512]
        output_dim = [256,512,512,512,512]
        for i in range(len(input_dim)):
            downblocks.append(DownBlock3d(input_dim[i], output_dim[i]))
        self.downblocks = nn.ModuleList(downblocks)

        upblocks = []
        input_dim = [512,1024,1024,1024,512]
        output_dim = [512,512,512,256,256]
        for i in range(len(input_dim)):
            upblocks.append(UpBlock3d(input_dim[i], output_dim[i]))
        
        self.reshape_depth = reshape_depth
        self.upblocks = nn.ModuleList(upblocks)
        self.first = SameBlock2d(3, self.reshape_depth*3,  kernel_size=(3, 3), padding=(1, 1))
        self.sameblock = SameBlock3d(256+num_kp+3, 256)

        self.second = SameBlock2d(self.reshape_depth*256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.third = SameBlock2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        # self.fourth = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        
        self.kp_variance = kp_variance
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(3, self.scale_factor)

    def create_heatmap_representations(self, spatial_size, kp):
        gaussian_driving = kp2gaussian(kp, spatial_size=spatial_size, kp_variance=self.kp_variance)
        return gaussian_driving
    
    def forward(self, keypoint, image, **kwargs):
        if self.scale_factor != 1:
            image = self.down(image)

        bs, c, h, w = image.shape
        spatial_size = (self.reshape_depth, h, w)
        keypoint_heatmap = self.create_heatmap_representations(spatial_size, keypoint)

        image = self.first(image)# (bs,3*depth,w,h)
        image_3d = image.view(bs, 3, self.reshape_depth, h ,w)
                
        out = torch.cat([keypoint_heatmap, image_3d], dim=1) # 3d的
        encoder_map = [out]
        for i in range(len(self.downblocks)):
            out = self.downblocks[i](out)
            encoder_map.append(out)
        out = encoder_map.pop()
        outs = []
        for i in range(len(self.upblocks)):
            out = self.upblocks[i](out)
            skip = encoder_map.pop()
            out = torch.cat([out, skip], dim=1)
        out = self.sameblock(out)
        bz, c, d,w,h = out.shape
        out = out.contiguous().view(bs, c*d, h, w)
        out =  self.second(out)
        out = self.third(out)
        return out


class AppearanceEncoder(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """
    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks,num_resblocks,reshape_depth=4,**kwargs):
        super(AppearanceEncoder, self).__init__()

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        
        down_blocks = []
        self.num_down_blocks = num_down_blocks
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features))
            
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.reshape_depth = reshape_depth
        self.reshape_channel = out_features//reshape_depth   

    def forward(self, source_image, **kwargs):
        # Encoding (downsampling) part
        out = self.first(source_image)
        encoder_map = [out] 

        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            encoder_map.append(out)

        return out, encoder_map # 2d, 2d feature_list

    def get_encode(self, driver_image):
        out = self.first(driver_image)
        encoder_map = []
        encoder_map.append(out)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out.detach())
            # out_mask = self.occlude_input(out.detach(), occlusion_map.detach())
            encoder_map.append(out.detach())
        
        return encoder_map
