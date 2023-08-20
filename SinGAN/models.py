import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import SinGAN.attention as attention

class Snake(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x * (1 / self.alpha) * torch.sin(x)* torch.sin(x)

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, opt, use_attn=False, generator=True):
        super(ConvBlock,self).__init__()
        if opt.spectral_norm_g and generator:
            print("Applying spectral norm to generator")
            self.add_module('conv',nn.utils.spectral_norm(nn.Conv3d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)))
        elif opt.spectral_norm_d and not generator:
            print("Applying spectral norm to discriminator")
            self.add_module('conv',nn.utils.spectral_norm(nn.Conv3d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)))
        else:    
            self.add_module('conv',nn.Conv3d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd))
        if opt.groupnorm:
            self.add_module('norm',nn.GroupNorm(min(32, out_channel // 2), out_channel))
        else:
            self.add_module('norm', nn.BatchNorm3d(out_channel))
        if not opt.prelu:
            self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))
        else:
            self.add_module('PRelu',nn.PReLU())
        if use_attn:
            if generator:
                self.add_module('CBAM', attention.CBAM(out_channel))
            else:
                self.add_module('CBAM', attention.CBAM(out_channel, no_spatial=opt.discrim_no_spatial))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1,opt, generator=False)
        self.body = nn.Sequential()
        num_layer = opt.num_layer_d if opt.num_layer_d else opt.num_layer
        for i in range(num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            if i == (num_layer - 2) // 2:
                block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1,opt,use_attn=opt.use_attention_d, generator=False)
            elif i == (num_layer - 2 - 1):
                block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1,opt,use_attn=opt.use_attention_end_d, generator=False)
            else:
                block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1,opt, generator=False)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv3d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x,feature_matching=False):
        x = self.head(x)
        feature = self.body(x)
        x = self.tail(feature)
        if feature_matching:
            return feature
        return x
    

class WDiscriminator_Branches(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator_Branches, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.branches = nn.ModuleList()
        NUM_BRANCHES = 6
        for _ in range(NUM_BRANCHES):
            N = int(opt.nfc)
            head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1,opt, generator=False)
            body = nn.Sequential()
            num_layer = opt.num_layer_d if opt.num_layer_d else opt.num_layer
            for i in range(num_layer-2):
                N = int(opt.nfc/pow(2,(i+1)))
                if i == (num_layer - 2) // 2:
                    block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1,opt,use_attn=opt.use_attention_d, generator=False)
                elif i == (num_layer - 2 - 1):
                    block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1,opt,use_attn=opt.use_attention_end_d, generator=False)
                else:
                    block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1,opt, generator=False)
                body.add_module('block%d'%(i+1),block)
            pre_tail = ConvBlock(max(N,opt.min_nfc),1,opt.ker_size,opt.padd_size,1,opt, generator=False)
            branch = nn.Sequential()
            branch.add_module('head',head)
            branch.add_module('body',body)
            branch.add_module('pre_tail',pre_tail)
            self.branches.append(branch)
        self.tail = nn.Conv3d(NUM_BRANCHES,1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x,feature_matching=False):
        original = x
        parts = []
        for branch in self.branches:
            parts.append(branch(original))
        parts = torch.cat(parts, dim=1).to(x.device)
        x = self.tail(parts)
        return x
    

class WDiscriminator_FocusedBranch(nn.Module):
    def __init__(self, opt):
        assert False, "Do not use"
        super(WDiscriminator_FocusedBranch, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.branches = nn.ModuleList()
        NUM_BRANCHES = 2
        for _ in range(NUM_BRANCHES):
            N = int(opt.nfc)
            head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1,opt, generator=False)
            body = nn.Sequential()
            num_layer = opt.num_layer_d if opt.num_layer_d else opt.num_layer
            for i in range(num_layer-2):
                N = int(opt.nfc/pow(2,(i+1)))
                if i == (num_layer - 2) // 2:
                    block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1,opt,use_attn=opt.use_attention_d, generator=False)
                elif i == (num_layer - 2 - 1):
                    block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1,opt,use_attn=opt.use_attention_end_d, generator=False)
                else:
                    block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1,opt, generator=False)
                body.add_module('block%d'%(i+1),block)
            branch = nn.Sequential()
            branch.add_module('head',head)
            branch.add_module('body',body)
            self.branches.append(branch)
        self.pre_tail = nn.Conv3d(NUM_BRANCHES*max(N,opt.min_nfc),max(N,opt.min_nfc),kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)
        self.tail = nn.Conv3d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x,feature_matching=False):
        assert x.shape[0] == 1 and x.shape[1] == 2
        full = self.branches[0](x[0,0,][None,None])
        focused = self.branches[1](x[0,1,][None,None])
        parts = torch.cat([full, focused], dim=1).to(x.device)
        x = self.pre_tail(parts)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1,opt) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            if i == (opt.num_layer - 2) // 2:
                block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1,opt,use_attn=opt.use_attention_g)
            elif i == (opt.num_layer - 2 - 1):
                block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1,opt,use_attn=opt.use_attention_end_g)
            else:
                block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1,opt)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv3d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )
        
    def forward(self,x,y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind),ind:(y.shape[4]-ind)]
        summed = x + y
        return summed
        
