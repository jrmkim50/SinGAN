import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import SinGAN.attention as attention
import SinGAN.functions as functions

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
        self.body = nn.ModuleList()
        num_layer = opt.num_layer_d if opt.num_layer_d else opt.num_layer
        for i in range(num_layer-2):
            N = int(D_NFC/pow(2,(i+1)))
            use_attn = False
            if i == (num_layer - 2) // 2:
                use_attn = opt.use_attention_d
            elif i == (num_layer - 2 - 1):
                use_attn = opt.use_attention_end_d
            block = convModule(max(2*N,opt.min_nfc)+(opt.nc_im if i < (num_layer - 2) // 2 else 0),max(N,opt.min_nfc),ker_size,paddD,1,opt,use_attn=use_attn, generator=False)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv3d(max(N,opt.min_nfc),1,kernel_size=ker_size,stride=1,padding=paddD)

    def forward(self,x):
        original = x
        x = self.head(x)
        for i, layer in enumerate(self.body):
            if i < len(self.body) // 2:
                ind = int(abs(x.shape[2]-original.shape[2])/2)
                original_ = original[:,:,ind:(original.shape[2]-ind),ind:(original.shape[3]-ind),ind:(original.shape[4]-ind)]
                x = torch.cat([x, original_], dim=1)
            x = layer(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        G_NFC = 2*opt.nfc if opt.doubleGFilters else opt.nfc
        G_MIN_NFC = 2*opt.min_nfc if opt.doubleGFilters else opt.min_nfc
        N = int(G_NFC)
        assert opt.num_layer == 6
        filters = [N, 2*N, 3*N, 2*N, N]
        convModule = ConvBlock if not opt.resnet else ConvRes
        ker_size = opt.ker_size
        paddG = opt.padd_size
        self.head = convModule(opt.nc_im,filters[0],ker_size,paddG,1,opt)
        self.body = nn.ModuleList()
        for i in range(opt.num_layer-2):
            N = int(G_NFC/pow(2,(i+1)))
            use_attn = False
            if i == (opt.num_layer - 2) // 2:
                use_attn = opt.use_attention_g
            elif i == (opt.num_layer - 2 - 1):
                use_attn = opt.use_attention_end_g
            block = convModule(filters[i]+(opt.nc_im if i < (opt.num_layer - 2) // 2 else 0),filters[i+1],ker_size,paddG,1,opt, use_attn=use_attn)
            self.body.add_module('block%d'%(i+1),block)
        if opt.finalConv:
            self.tail = nn.Sequential(
                nn.Conv3d(max(N,G_MIN_NFC),opt.nc_im,kernel_size=ker_size,stride =1,padding=paddG),
                nn.BatchNorm3d(opt.nc_im),
                nn.Tanh()
            )
            self.output = nn.Sequential(
                nn.Conv3d(opt.nc_im,opt.nc_im,kernel_size=1),
                nn.Tanh()
            )
        else:
            self.tail = nn.Sequential(
                nn.Conv3d(max(N,G_MIN_NFC),opt.nc_im,kernel_size=ker_size,stride =1,padding=paddG),
                nn.Tanh()
            )
            self.output = nn.Identity()


        
    def forward(self,x,y):
        original_noise = x
        x = self.head(x)
        for i, layer in enumerate(self.body):
            if i < len(self.body) // 2:
                ind = int(abs(x.shape[2]-original_noise.shape[2])/2)
                original_ = original_noise[:,:,ind:(original_noise.shape[2]-ind),ind:(original_noise.shape[3]-ind),ind:(original_noise.shape[4]-ind)]
                x = torch.cat([x, original_], dim=1)
            x = layer(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind),ind:(y.shape[4]-ind)]
        summed = x + y
        summed = self.output(summed)
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


from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import random
class StageDataModule(Dataset):
    def __init__(
        self,
        batch_sizes: Tuple[int, int], # bs_gen, bs_recon
        noise_shape_info: Tuple[int,Tuple], # nc_z, scale0_size--the (x,y,z) portion of the shape
        real_and_extra,
        scale_info: Tuple[any, int], # z_opt3D_FULL, scale_num
        noise_padding_module,
        opt
    ):
        super().__init__()
        # Set bs
        self.bs_gen, self.bs_recon = batch_sizes
        # Info about real imgs
        self.real_and_extra = real_and_extra
        self.total_samps = len(real_and_extra)
        # base noise vectors
        nc_z, scale0_size = noise_shape_info
        self.nc_z = nc_z
        self.in_s = torch.zeros(self.bs_gen, self.nc_z, *scale0_size)
        self.in_s_z_opt = torch.zeros(self.bs_recon, self.nc_z, *scale0_size)
        # data for curr scale
        self.z_opt3D_FULL, self.scale_num = scale_info
        # padding module
        self.noise_padding_module = noise_padding_module
        self.len = opt.Gsteps + opt.Dsteps
        
    
    def __len__(self):
        return self.len

    def __getitem__(self, _idx):
        # data for gen with batch size bs_gen
        real_size = self.real_and_extra.shape[2:] # b,c,w,h,d
        img_indices_for_gen = random.sample(range(self.total_samps),self.bs_gen)
        real_imgs = [self.real_and_extra[idx][None] for idx in img_indices_for_gen]
        real_imgs = torch.cat(real_imgs)
        if self.scale_num == 0:
            noise_3D = functions.generate_noise3D([1,*real_size],device='cpu',num_samp=self.bs_gen)
            noise_3D = self.noise_padding_module(noise_3D.expand(self.bs_gen,self.nc_z,*real_size))
        else:
            noise_3D = functions.generate_noise3D([self.nc_z,*real_size], device='cpu',num_samp=self.bs_gen)
            noise_3D = self.noise_padding_module(noise_3D)
        # data for recon with batch size bs_recon
        img_indices_for_recon = random.sample(range(self.total_samps),self.bs_recon)
        recon_imgs = [self.real_and_extra[idx][None] for idx in img_indices_for_recon]
        recon_imgs = torch.cat(recon_imgs)
        z_opt3D = [self.z_opt3D_FULL[idx][None] for idx in img_indices_for_recon]
        z_opt3D = torch.cat(z_opt3D)
        # real_imgs comes out normalized between -1 and 1
        return real_imgs, noise_3D, self.in_s, self.in_s_z_opt, z_opt3D, recon_imgs, img_indices_for_recon


import pytorch_lightning as pl
from torchmetrics.functional import structural_similarity_index_measure as ssim
from collections import OrderedDict
import matplotlib.pyplot as plt
import nibabel as nib
import os
class SinGAN(pl.LightningModule):

    def __init__(self, G, D, Gs, Zs, NoiseAmp, 
                 padding_modules: Tuple, # (m_noise, m_image)
                 reals3D, real_and_extra, bs: Tuple[int, int],
                 opt):
        super().__init__()

        self.generator = G
        self.discriminator = D
        self.Gs = Gs
        self.Zs = Zs
        self.NoiseAmp = NoiseAmp
        self.m_noise3D, self.m_image3D = padding_modules
        self.bs_gen, self.bs_recon = bs
        self.reals3D = reals3D
        self.real_and_extra = real_and_extra
        if len(Gs) == 0:
            scale_size = self.reals3D[len(self.Gs)].shape[2:]
            total_samps = len(self.real_and_extra)
            self.z_opt3D_FULL = functions.generate_noise3D([1,*scale_size], device='cpu', num_samp=total_samps)
            self.z_opt3D_FULL = self.m_noise3D(self.z_opt3D_FULL.expand(total_samps,opt.nc_z,*scale_size))
        else:
            self.z_opt3D_FULL = self.m_noise3D(torch.full(self.real_and_extra.shape, 0, device='cpu'))
        self.opt = opt

    def forward(self, noisy, prev):
        return self.generator(noisy, prev)
    
    def sim_loss(self, fake, real):
        return -ssim(fake, real)

    def _draw_concat3D(self,in_s,mode,recon_idxs=None):
        G_z = in_s
        if len(self.Gs) > 0:
            if mode == 'rand':
                count = 0
                for G,real_curr,real_next,noise_amp in zip(self.Gs,self.reals3D,self.reals3D[1:],self.NoiseAmp):
                    # make noise for curr scale: we make bs_gen noise instances
                    if count == 0:
                        noise_shape = [1, real_curr.shape[2], real_curr.shape[3], real_curr.shape[4]]
                        z3D = functions.generate_noise3D(noise_shape, device=self.device, num_samp=self.bs_gen)
                        z3D = z3D.expand(self.bs_gen, self.opt.nc_z, *z3D.shape[2:])
                    else:
                        noise_shape = [self.opt.nc_z, real_curr.shape[2], real_curr.shape[3], real_curr.shape[4]]
                        z3D = functions.generate_noise3D(noise_shape, device=self.device, num_samp=self.bs_gen)
                    z3D = self.m_noise3D(z3D)
                    # Crop prev output to right size
                    G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3],0:real_curr.shape[4]]
                    G_z = self.m_image3D(G_z)
                    # Create noisy image for curr scale to generate from
                    z_in = noise_amp*z3D+G_z
                    G_z = G(z_in.detach(),G_z)
                    # Resize for next scale
                    G_z = functions.imresize3D_pl(G_z,1/self.opt.scale_factor,self.device)
                    G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3],0:real_next.shape[4]]
                    count += 1
            if mode == 'rec':
                count = 0
                for G,Z_opt,real_curr,real_next,noise_amp in zip(self.Gs,self.Zs,self.reals3D,self.reals3D[1:],self.NoiseAmp):
                    # Crop prev output to right size
                    G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3], 0:real_curr.shape[4]]
                    G_z = self.m_image3D(G_z)
                    # Set noise to correspond to the correct recon images
                    Z_opt = [Z_opt[idx][None] for idx in recon_idxs]
                    Z_opt = torch.cat(Z_opt)
                    z_in = noise_amp*Z_opt+G_z
                    G_z = G(z_in.detach(),G_z)
                    G_z = functions.imresize3D_pl(G_z,1/self.opt.scale_factor,self.device)
                    G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3],0:real_next.shape[4]]
                    #if count != (len(Gs)-1):
                    #    G_z = m_image(G_z)
                    count += 1
        return G_z
    
    def _log_data(self, fake, opt_images, z_prev3D, recon_img_data):
        recon_imgs, recon_idxs = recon_img_data
        plt.imsave('%s/fake_sample.png' %  (self.opt.outf), functions.convert_image_np3D(fake, opt=self.opt), vmin=0, vmax=1)
        for idx, opt_img in enumerate(opt_images):
            img_idx = recon_idxs[idx]
            plt.imsave('%s/z_prev_%d.png'    % (self.opt.outf, img_idx),  functions.convert_image_np3D(z_prev3D[idx][None], opt=self.opt), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt)_%d.png'    % (self.opt.outf, img_idx),  functions.convert_image_np3D(opt_img[None], opt=self.opt), vmin=0, vmax=1)
            plt.imsave('%s/real_%d.png'    % (self.opt.outf, img_idx),  functions.convert_image_np3D(recon_imgs[idx][None], opt=self.opt), vmin=0, vmax=1)
            for name, img in [("z_prev", z_prev3D[idx][None]), ("opt_img", opt_img[None]), ("real", recon_imgs[idx][None])]:
                to_save = functions.convert_image_np3D(img, eval=True, opt=self.opt)
                img = nib.Nifti1Image(to_save[:,:,:,:], np.eye(4))
                nib.save(img, os.path.join(self.opt.outf, f"{name}_{img_idx}.nii.gz"))
        torch.save(self.z_opt3D_FULL, '%s/z_opt.pth' % (self.opt.outf))

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, noise_3D, in_s, in_s_z_opt, z_opt3D, recon_imgs, recon_idxs = batch
        prev_gen = self.m_image3D(self._draw_concat3D(in_s,'rand',recon_idxs=None))
        if self.Gs == []:
            noise3D = noise_3D
        else:
            noise3D = self.opt.noise_amp*noise_3D+prev_gen

        if optimizer_idx == 0: # generator loop
            # make the fake image and optimize it
            fake = self(noise3D.detach(),prev_gen)
            output = self.discriminator(fake)
            errG = -output.mean()
            should_compute_sim = (self.opt.sim_alpha != 0 and 
                                ((self.opt.sim_boundary_type == "start" and len(self.Gs) >= self.opt.sim_boundary) or 
                                (self.opt.sim_boundary_type == "end" and len(self.Gs) <= self.opt.sim_boundary)))
            if should_compute_sim:
                fake_adjusted = (fake + 1) / 2
                real_adjusted = (real_imgs + 1) / 2
                assert fake_adjusted.shape == real_adjusted.shape
                ssim_loss = self.sim_loss(fake_adjusted, real_adjusted)
                ssim_loss = self.opt.sim_alpha * ssim_loss
                errG += ssim_loss
            
            loss = nn.L1Loss()
            if self.Gs == []:
                z_prev3D = self.m_noise3D(torch.zeros_like(in_s_z_opt))
            else:
                z_prev3D = self.m_noise3D(self._draw_concat3D(in_s_z_opt,'rec',recon_idxs=recon_idxs))
            assert z_opt3D.shape == z_prev3D.shape
            Z_opt = self.opt.noise_amp*z_opt3D+z_prev3D
            fake_recon = self(Z_opt.detach(),z_prev3D)
            rec_loss = self.opt.alpha*loss(fake_recon, recon_imgs)
            errG += rec_loss

            # focused_loss = nn.L1Loss()
            # fake_recon_highs = fake_recon.max(dim=3).values
            # focused_rec_loss = self.opt.alpha // 2 * focused_loss(fake_recon_highs, recon_imgs.max(dim=3).values)
            # errG += focused_rec_loss

            # focused_loss = nn.L1Loss(reduction='sum')
            # zeroes = torch.zeros_like(fake_recon)
            # zeroes[:,:,:,random.choice(list(range(fake_recon.shape[3])))] = 1
            # alphas = [1/300, 1/400, 1/500, 1/800, 1/1200, 1/2000, 1/3000]
            # focused_rec_loss = alphas[len(self.Gs)]*focused_loss(fake_recon * zeroes, recon_imgs * zeroes)
            # errG += focused_rec_loss

            # report loss
            tqdm_dict = {"g_loss": errG.item(), "rec_loss": rec_loss.item(), 'lr': self.trainer.optimizers[optimizer_idx].param_groups[0]['lr']}
            output = OrderedDict({"loss": errG, "progress_bar": tqdm_dict})
            # log tqdm dict
            if ((self.current_epoch % 25 == 0 or self.current_epoch == self.trainer.max_epochs - 1) 
                and batch_idx == self.opt.Gsteps - 1):
                print(f"scale {len(self.Gs)}: {self.current_epoch}/{self.trainer.max_epochs}", tqdm_dict)
            # log images
            if ((self.current_epoch % 500 == 0 or self.current_epoch == self.trainer.max_epochs - 1) 
                and batch_idx == self.opt.Gsteps - 1):
                self._log_data(fake.detach(), fake_recon.detach(), z_prev3D, (recon_imgs, recon_idxs))
            return output
        elif optimizer_idx == 1: # discriminator loop
            # train on real
            output_real = self.discriminator(real_imgs)
            errD_real = -output_real.mean()
            # train on fake
            fake = self(noise3D.detach(),prev_gen)
            assert fake.shape == real_imgs.shape
            output_fake = self.discriminator(fake.detach())
            errD_fake = output_fake.mean()
            # gp
            gradient_penalty = functions.calc_gradient_penalty(self.discriminator, real_imgs, fake, self.opt.lambda_grad, self.device)
            # report loss
            d_loss = errD_real + errD_fake + gradient_penalty
            tqdm_dict = {'d_loss': d_loss.item(), 'fake_score': output_fake.mean().item(), 'real_score': output_real.mean().item(), 
                         'lr': self.trainer.optimizers[optimizer_idx].param_groups[0]['lr']}
            if ((self.current_epoch % 25 == 0 or self.current_epoch == self.trainer.max_epochs - 1) 
                and batch_idx == self.opt.Gsteps + self.opt.Dsteps - 1):
                print(f"scale {len(self.Gs)}: {self.current_epoch}/{self.trainer.max_epochs}", tqdm_dict)
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
            })
            return output

    def configure_optimizers(self):
        def lr(epoch):
            # 0: 0.1^5, 1: 0.1^4, 2: 0.1^3, 3: 0.1^2, 4: 0.1^1, ..., WARMUP_EPOCHS: 0.1^0, 
            # WARMUP_EPOCHS+1: decayLR^1, ...
            WARMUP_EPOCHS = 5
            decayLR = self.opt.gamma**(1/(0.8*self.opt.niter-WARMUP_EPOCHS))
            if epoch <= WARMUP_EPOCHS:
                lr_scale = 0.1**(WARMUP_EPOCHS-epoch)
            else:
                # Calculated so that at epoch 1600, we are multiplying lr by 0.1 (opt.gamma)
                lr_scale = decayLR**(epoch-WARMUP_EPOCHS)
                # lr_scale = 1-((epoch - 5) / 2000)
            return lr_scale

        optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr_g, betas=(self.opt.beta1, 0.999))
        optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr_d, betas=(self.opt.beta1, 0.999))
        
        if self.opt.warmup_d:
            schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lr)
        else:
            schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[0.8*self.opt.niter],gamma=self.opt.gamma)
        
        if self.opt.warmup_g:
            schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lr)
        else:
            schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[0.8*self.opt.niter],gamma=self.opt.gamma)

        return (
            {'optimizer': optimizerG, 'frequency': self.opt.Gsteps, 'lr_scheduler': { "scheduler": schedulerG, "interval": "step", "frequency": self.opt.Gsteps }},
            {'optimizer': optimizerD, 'frequency': self.opt.Dsteps, 'lr_scheduler': { "scheduler": schedulerD, "interval": "step", "frequency": self.opt.Dsteps }}
        )

    def train_dataloader(self):
        dataset = StageDataModule((self.bs_gen, self.bs_recon), (self.opt.nc_z, self.reals3D[0].shape[2:]), 
                                  self.real_and_extra, (self.z_opt3D_FULL, len(self.Gs)), self.m_noise3D, self.opt)
        return DataLoader(dataset, batch_size=None)