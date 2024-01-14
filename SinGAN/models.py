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
            self.add_module('norm',nn.GroupNorm(8, out_channel))
        elif opt.instanceNorm:
            self.add_module('norm', nn.InstanceNorm3d(out_channel, affine=True))
        else:
            if opt.instanceNormD and not generator:
                self.add_module('norm', nn.InstanceNorm3d(out_channel, affine=True))
            else:
                self.add_module('norm', nn.BatchNorm3d(out_channel))
        if not opt.prelu:
            if generator and opt.reluG:
                self.add_module('Relu',nn.ReLU(inplace=True))
            else:    
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
    if isinstance(m, nn.Conv3d):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        D_NFC = opt.nfc if not opt.doubleDFilters else 2*opt.nfc
        N = int(D_NFC)
        paddD = 0 if opt.noPadD else opt.padd_size
        convModule = ConvBlock if not opt.resnet else ConvRes
        ker_size = opt.ker_size_d if not opt.planarD else (opt.ker_size_d, 1, opt.ker_size_d)
        self.head = convModule(opt.nc_im,N,ker_size,paddD,1,opt, generator=False)
        self.body = nn.Sequential()
        num_layer = opt.num_layer_d if opt.num_layer_d else opt.num_layer
        for i in range(num_layer-2):
            N = int(D_NFC/pow(2,(i+1)))
            use_attn = False
            if i == (num_layer - 2) // 2:
                use_attn = opt.use_attention_d
            elif i == (num_layer - 2 - 1):
                use_attn = opt.use_attention_end_d
            block = convModule(max(2*N,opt.min_nfc),max(N,opt.min_nfc),ker_size,paddD,1,opt,use_attn=use_attn, generator=False)
            self.body.add_module('block%d'%(i+1),block)
        self.skip = None if not opt.skipD else ConvBlock(max(N,opt.min_nfc)+int(D_NFC), max(N,opt.min_nfc),ker_size,paddD,1,opt, generator=False)
        self.tail = nn.Conv3d(max(N,opt.min_nfc),1,kernel_size=ker_size,stride=1,padding=paddD)

    def forward(self,x):
        x = self.head(x)
        feature = self.body(x)
        if self.skip:
            ind = int((x.shape[2]-feature.shape[2])/2)
            x = x[:,:,ind:(x.shape[2]-ind),ind:(x.shape[3]-ind),ind:(x.shape[4]-ind)]
            feature = torch.cat([x, feature], dim=1)
            feature = self.skip(feature)
        x = self.tail(feature)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        G_NFC = 2*opt.nfc if opt.doubleGFilters else opt.nfc
        G_MIN_NFC = 2*opt.min_nfc if opt.doubleGFilters else opt.min_nfc
        N = int(G_NFC)
        convModule = ConvBlock if not opt.resnet else ConvRes
        self.head = convModule(opt.nc_im * 2 if opt.combineWithR else opt.nc_im,N,opt.ker_size,opt.padd_size,1,opt)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(G_NFC/pow(2,(i+1)))
            use_attn = False
            if i == (opt.num_layer - 2) // 2:
                use_attn = opt.use_attention_g
            elif i == (opt.num_layer - 2 - 1):
                use_attn = opt.use_attention_end_g
            block = convModule(max(2*N,G_MIN_NFC),max(N,G_MIN_NFC),opt.ker_size,opt.padd_size,1,opt, use_attn=use_attn)
            self.body.add_module('block%d'%(i+1),block)
        # pad the skip convolution so we don't worry about shapes
        self.skip = None if not opt.skipG else ConvBlock(max(N,G_MIN_NFC)+int(G_NFC), max(N,G_MIN_NFC),opt.ker_size,opt.ker_size//2,1,opt)
        if opt.finalConv:
            self.tail = nn.Sequential(
                nn.Conv3d(max(N,G_MIN_NFC),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
                nn.BatchNorm3d(opt.nc_im),
                nn.Tanh()
            )
            self.output = nn.Sequential(
                nn.Conv3d(opt.nc_im,opt.nc_im,kernel_size=1),
                nn.Tanh()
            )
        else:
            self.tail = nn.Sequential(
                nn.Conv3d(max(N,G_MIN_NFC),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
                nn.Tanh()
            )
            self.output = nn.Identity()
        self.resnet = None
        if opt.resnetV2G:
            assert not opt.finalConv and not opt.resnet
            resLayers = []
            N = int(G_NFC)
            resLayers.append(convModule(opt.nc_im, N, opt.ker_size, opt.ker_size // 2, 1, opt))
            for i in range(opt.num_layer - 2):
                N = int(G_NFC/pow(2,(i+1)))
                use_attn = False
                if i == (opt.num_layer - 2) // 2:
                    use_attn = opt.use_attention_g
                elif i == (opt.num_layer - 2 - 1):
                    use_attn = opt.use_attention_end_g
                block = convModule(max(2*N,G_MIN_NFC),max(N,G_MIN_NFC),opt.ker_size,opt.ker_size // 2,1,opt, use_attn=use_attn)
                resLayers.append(block)
            resLayers.append(nn.Sequential(
                nn.Conv3d(max(N,G_MIN_NFC),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.ker_size // 2),
                nn.Tanh()
            ))
            self.resnet = nn.Sequential(*resLayers)


        
    def forward(self,x,y):
        head = self.head(x)
        x = self.body(head)
        if self.skip:
            ind = int((head.shape[2]-x.shape[2])/2)
            head = head[:,:,ind:(head.shape[2]-ind),ind:(head.shape[3]-ind),ind:(head.shape[4]-ind)]
            x = torch.cat([head, x], dim=1)
            x = self.skip(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind),ind:(y.shape[4]-ind)]
        summed = x + y
        summed = self.output(summed)
        if self.resnet:
            residual = self.resnet(summed)
            return summed + residual
        return summed
    

class encoderBlock(nn.Module):
    def __init__(self, in_c, out_c, ker_size, padd, stride, opt, use_attn=False, generator=True):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c, ker_size, padd, stride, opt, use_attn, generator)
        self.pool = nn.MaxPool3d(2)
    
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
    
class decoderBlock(nn.Module):
    def __init__(self, in_c, out_c, ker_size, padd, stride, opt, use_attn=False, generator=True):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_c, out_c, kernel_size=ker_size,padding=ker_size//2)
        )
        self.conv = ConvBlock(out_c+out_c, out_c, ker_size, padd, stride, opt, use_attn, generator)
    
    def forward(self, inputs, skip):
        x = self.up(inputs)
        diffZ = skip.size()[2] - x.size()[2]
        diffY = skip.size()[3] - x.size()[3]
        diffX = skip.size()[4] - x.size()[4]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2, 
                      diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self, opt, is_generator):
        super().__init__()
        # input size == output size
        self.is_generator = is_generator
        N = opt.nfc
        padd = opt.ker_size // 2
        self.e1 = encoderBlock(opt.nc_im, N, opt.ker_size, padd, 1, opt, generator=is_generator)
        self.e2 = encoderBlock(N, 2*N, opt.ker_size, padd, 1, opt, generator=is_generator)
        self.e3 = encoderBlock(2*N, 2*N, opt.ker_size, padd, 1, opt, generator=is_generator)
        
        self.b = ConvBlock(2*N, 2*N, opt.ker_size, padd, 1, opt, generator=is_generator,use_attn=opt.use_attention_g if is_generator else opt.use_attention_d)

        self.d1 = decoderBlock(2*N, 2*N, opt.ker_size, padd, 1, opt, generator=is_generator)
        self.d2 = decoderBlock(2*N, 2*N, opt.ker_size, padd, 1, opt, generator=is_generator,use_attn=opt.use_attention_end_g if is_generator else opt.use_attention_end_d)
        self.d3 = decoderBlock(2*N, N, opt.ker_size, padd, 1, opt, generator=is_generator)

        self.outputs = nn.Sequential(
            nn.Conv3d(N, opt.nc_im, kernel_size=1, padding=0),
            nn.Tanh()
        ) if is_generator else nn.Conv3d(N, 1, kernel_size=1, padding=0)
    
    def forward(self, x, y = None):
        
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b = self.b(p3)

        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        x = self.outputs(d3)
        # TODO 11/18: check whether this summing helps?? Right now, keeping the summing
        if not self.is_generator:
            return x
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind),ind:(y.shape[4]-ind)]
        summed = x + y
        return summed

class ConvRes(nn.Module):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, opt, use_attn=False, generator=True):
        super().__init__()
        self.conv = ConvBlock(in_channel, out_channel, ker_size, padd, stride, opt, use_attn, generator)
        self.res = ResBlock(out_channel, ker_size, use_groupnorm=False)
    def forward(self, x):
        return self.res(self.conv(x))

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        use_groupnorm = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels) if use_groupnorm else nn.BatchNorm3d(in_channels)
        self.norm2 = nn.GroupNorm(8, in_channels) if use_groupnorm else nn.BatchNorm3d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = Convolution(
            in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False
        )
        self.conv2 = Convolution(
            in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False
        )

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x

class Convolution(nn.Module):
    # a hack to align with monai
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias, stride=1):
        super(Convolution, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, stride=stride)

    def forward(self, x):
        return self.conv(x)

class FinetuneNet(nn.Module):

    def __init__(self, opt):
        super(FinetuneNet, self).__init__()
        freezeOut = 32
        self.head = ConvBlock(opt.nc_im, 1, 3, 1, 1, opt, generator=False)

        # self.down_layers and self.convInit are initialized by the pretrained model
        self.convInit = Convolution(1, 32, 3, 1, False)
        self.down_layers = nn.ModuleList([nn.Sequential(nn.Identity(), ResBlock(32, 3))])
        assert opt.pretrainD <= 2
        if opt.pretrainD == 2:
            self.down_layers.append(nn.Sequential(Convolution(32, 64, 3, 1, False, stride=2), ResBlock(64), ResBlock(64)))
            freezeOut = 64

        D_NFC = opt.nfc if not opt.doubleDFilters else 2*opt.nfc
        N = int(D_NFC)
        paddD = 0 if opt.noPadD else opt.padd_size
        convModule = ConvBlock if not opt.resnet else ConvRes
        self.body = nn.Sequential()
        num_layer = opt.num_layer_d if opt.num_layer_d else opt.num_layer
        for i in range(num_layer-opt.pretrainD-2):
            N = int(D_NFC/pow(2,(i+1)))
            use_attn = False
            if i == (num_layer - 2) // 2:
                use_attn = opt.use_attention_d
            elif i == (num_layer - 2 - 1):
                use_attn = opt.use_attention_end_d
            block = convModule(max(2*N,opt.min_nfc) if i > 0 else freezeOut,max(N,opt.min_nfc),opt.ker_size_d,paddD,1,opt, use_attn=use_attn, generator=False)
            self.body.add_module('block%d'%(i+1),block)
        # skip takes +1 for in_channels because the head outputs 1
        self.skip = None if not opt.skipD else ConvBlock(max(N,opt.min_nfc)+1, max(N,opt.min_nfc),opt.ker_size_d,paddD,1,opt, generator=False)
        self.tail = nn.Conv3d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size_d,stride=1,padding=paddD)

    def forward(self,x):
        x = self.head(x)
        convInit = self.convInit(x)
        feature = self.down_layers[0](convInit)
        if len(self.down_layers) == 2:
            feature = self.down_layers[1](feature)
        feature = self.body(feature)
        if self.skip:
            ind = int((x.shape[2]-feature.shape[2])/2)
            x = x[:,:,ind:(x.shape[2]-ind),ind:(x.shape[3]-ind),ind:(x.shape[4]-ind)]
            feature = torch.cat([x, feature], dim=1)
            feature = self.skip(feature)
        x = self.tail(feature)
        return x
