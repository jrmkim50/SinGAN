import SinGAN.functions as functions
import SinGAN.models as models
import SinGAN.models_2d as models_2d
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize, imresize3D
from torchmetrics.functional import structural_similarity_index_measure as ssim
from SinGAN.perceptual import VGGLoss
from SinGAN.perceptual_3D import MedicalNetLoss
import random
import nibabel as nib
import numpy as np

def train(opt,Gs,Zs,reals,NoiseAmp):

    real_, extra_images = functions.read_image3D(opt)
    for extra_image in extra_images:
        assert extra_image.shape == real_.shape
    scale_num = 0
    real = imresize3D(real_,opt.scale1,opt)
    for idx in range(len(extra_images)):
        # Resize each extra image to fit the original scale
        extra_images[idx] = imresize3D(extra_images[idx], opt.scale1, opt)
    for extra_image in extra_images:
        assert extra_image.shape == real.shape
    reals = functions.creat_reals_pyramid3D(real,reals,opt)
    extra_pyramids = []
    for image in extra_images:
        extra_pyramids.append(functions.creat_reals_pyramid3D(image, [], opt))
    for pyramid in extra_pyramids:
        assert len(pyramid) == len(reals)
        for pyramid_level in range(len(pyramid)):
            assert pyramid[pyramid_level].shape == reals[pyramid_level].shape
            assert abs(pyramid[pyramid_level]-reals[pyramid_level]).sum() != 0
    nfc_prev = 0
    print("Num layers:", opt.stop_scale)
    in_s_z_opt_FULL = torch.zeros_like(reals[0]).expand(len(extra_images) + 1, *reals[0].shape[1:]).to(opt.device)

    while scale_num<opt.stop_scale+1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        if scale_num > 0 and opt.growD and scale_num % opt.growD == 0:
            # 21, 25, 30, 36, 44, 53, 64
            # growD=3 goes for 4: 9, 5: 11, 6: 13. growD=4 explores 4 and 5.
            opt.num_layer_d += 1

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass

        #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np3D(reals[scale_num], opt=opt), vmin=0, vmax=1)

        D_curr,G_curr = init_models(opt, reals[scale_num].shape, scale_num)
        if nfc_prev==opt.nfc:
            print("loading gen")
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            print("loading discrim")
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)), strict=False)

        z_curr,G_curr,D_curr = train_single_scale3D(D_curr,G_curr,reals,extra_pyramids,Gs,Zs,in_s_z_opt_FULL,NoiseAmp,opt)

        assert in_s_z_opt_FULL.shape[1:] == reals[0].shape[1:], "in_s_z_opt is always same shape as first scale"

        G_curr = functions.reset_grads(G_curr,False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr,False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr,G_curr
    return extra_pyramids

class SimLoss(nn.Module):
    def __init__(self, use_harmonic):
        self.use_harmonic = use_harmonic
        super(SimLoss, self).__init__()

    def forward(self, fake, real):
        if self.use_harmonic:
            return 1-ssim(fake, real, reduction='none')
        return -1 * ssim(fake, real)

def harmonic_mean(nums):
    assert len(nums) > 0
    return len(nums) / torch.reciprocal(nums + 1e-16).sum()

import pytorch_lightning as pl

def train_single_scale3D(netD,netG,reals3D,extra_pyramids,Gs,Zs,in_s_z_opt_FULL,NoiseAmp,opt,centers=None):
    assert opt.sim_type == "ssim", "ssim option only"
    real = reals3D[len(Gs)]
    to_cat = [real,]
    to_cat += [pyramid[len(Gs)] for pyramid in extra_pyramids]
    for im in to_cat:
        assert im.shape == real.shape
    real_and_extra = torch.cat(to_cat).to(opt.device)
    opt.nzx = real.shape[2]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real.shape[3]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzz = real.shape[4]#+(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2) if (not opt.unetG and opt.padd_size == 0) else 0
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2) if (not opt.unetG and opt.padd_size == 0) else 0
    m_noise3D = nn.ConstantPad3d(int(pad_noise), 0)
    m_image3D = nn.ConstantPad3d(int(pad_image), 0)

    niter = opt.niter

    if opt.train_last_layer_longer and len(Gs) == opt.stop_scale:
        print("Training last layer for longer")
        niter *= 2
    elif opt.train_first_layers_longer and len(Gs) < opt.train_first_layers_longer:
        print("Training layer for longer")
        niter *= 2

    if (Gs == []) & (opt.mode != 'SR_train'):
        opt.noise_amp = 1
    else:
        _z_prev3D_FULL = draw_concat3D(Gs,Zs,reals3D,NoiseAmp,in_s_z_opt_FULL,'rec',m_noise3D,m_image3D,opt)
        criterion = nn.MSELoss()
        assert _z_prev3D_FULL.shape == real_and_extra.shape, f"{_z_prev3D_FULL.shape} versus {real_and_extra.shape}"
        RMSE = torch.sqrt(criterion(real_and_extra, _z_prev3D_FULL))
        opt.noise_amp = opt.noise_amp_init*RMSE

    bs = (3, 3) # (3, 3)
    model = models.SinGAN(netG, netD, Gs, Zs, NoiseAmp, (m_noise3D, m_image3D), reals3D, real_and_extra, bs, opt)
    trainer = pl.Trainer(max_epochs=niter, gpus=[opt.device.index], 
                         logger=None, enable_checkpointing=False, enable_progress_bar=False)
    trainer.fit(model)

    netG = netG.to(opt.device)
    model.z_opt3D_FULL = model.z_opt3D_FULL.to(opt.device)

    functions.save_networks(netG,netD,model.z_opt3D_FULL,opt)
    # if netD_fine:
    #     torch.save(netD_fine.state_dict(), '%s/netD_fine.pth' % (opt.outf))
    return model.z_opt3D_FULL,netG,netD 

def draw_concat3D(Gs,Zs,reals3D,NoiseAmp,in_s,mode,m_noise3D,m_image3D,opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2) if (not opt.unetG and opt.padd_size == 0) else 0
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals3D,reals3D[1:],NoiseAmp):
                if count == 0:
                    noise_shape = [1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise, Z_opt.shape[4] - 2 * pad_noise]
                    z3D = functions.generate_noise3D(noise_shape, device=opt.device)
                    z3D = z3D.expand(1, opt.nc_z, z3D.shape[2], z3D.shape[3], z3D.shape[4])
                else:
                    noise_shape = [opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise, Z_opt.shape[4] - 2 * pad_noise]
                    z3D = functions.generate_noise3D(noise_shape, device=opt.device)
                z3D = m_noise3D(z3D)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3],0:real_curr.shape[4]]
                G_z = m_image3D(G_z)
                z_in = noise_amp*z3D+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize3D(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3],0:real_next.shape[4]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals3D,reals3D[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3], 0:real_curr.shape[4]]
                G_z = m_image3D(G_z)
                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize3D(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3],0:real_next.shape[4]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z

def train_paint(opt,Gs,Zs,reals,NoiseAmp,centers,paint_inject_scale):
    assert False, "not implemented"
    in_s = torch.full(reals[0].shape, 0, device=opt.device)
    scale_num = 0
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        if scale_num!=paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_,scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                    pass

            #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
            #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/in_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr,G_curr = init_models(opt)

            z_curr,in_s,G_curr = train_single_scale3D(D_curr,G_curr,reals[:scale_num+1],Gs[:scale_num],Zs[:scale_num],in_s,NoiseAmp[:scale_num],opt,centers=centers)

            G_curr = functions.reset_grads(G_curr,False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr,False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num+=1
            nfc_prev = opt.nfc
        del D_curr,G_curr
    return


def init_models(opt, real_shape, scale_num):

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device) if not opt.unetG else models.Unet(opt, True).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device) if not opt.unetD else models.Unet(opt, False).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG