import SinGAN.functions as functions
import SinGAN.models as models
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
import random
import nibabel as nib
import numpy as np

def train(opt,Gs,Ds,Zs,reals,NoiseAmp):

    # DISABLING THESE OPTIONS FOR NOW
    assert not opt.generate_with_critic
    assert not opt.sim_cond_d
    assert not opt.planar_convs

    real_, extra_images = functions.read_image3D(opt)
    for extra_image in extra_images:
        assert extra_image.shape == real_.shape
    in_s = 0
    in_s_z_opt = 0
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

    while scale_num<opt.stop_scale+1:
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
        plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np3D(reals[scale_num], opt=opt), vmin=0, vmax=1)

        D_curr,G_curr = init_models(opt, scale_num)
        if (nfc_prev==opt.nfc) and (not opt.generate_with_critic or scale_num > 1):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))

        z_curr,in_s,in_s_z_opt,G_curr,D_curr = train_single_scale3D(D_curr,G_curr,reals,extra_pyramids,Gs,Ds,Zs,in_s,in_s_z_opt,NoiseAmp,opt)

        G_curr = functions.reset_grads(G_curr,False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr,False)
        D_curr.eval()

        Gs.append(G_curr)
        Ds.append(D_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        if opt.generate_with_critic:
            torch.save(Ds, '%s/Ds.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr,G_curr
    return

class TargetSimLoss(nn.Module):
    def __init__(self, target, ssim):
        super(TargetSimLoss, self).__init__()
        self.target = target
        self.ssim = ssim

    def forward(self, fake, real):
        return torch.abs(self.ssim(fake, real) - self.target)
    
class SimLoss(nn.Module):
    def __init__(self, use_harmonic):
        self.use_harmonic = use_harmonic
        super(SimLoss, self).__init__()

    def forward(self, fake, real):
        if self.use_harmonic:
            return 1-ssim(fake, real, reduction='none')
        return -1 * ssim(fake, real)

def harmonic_mean(nums):
    assert len(nums) == 3
    return len(nums) / torch.reciprocal(nums + 1e-16).sum()

def train_single_scale3D(netD,netG,reals3D,extra_pyramids,Gs,Ds,Zs,in_s,in_s_z_opt,NoiseAmp,opt,centers=None):
    real = reals3D[len(Gs)]
    to_cat = [real,]
    to_cat += [pyramid[len(Gs)] for pyramid in extra_pyramids]
    for im in to_cat:
        assert im.shape == real.shape
    real_and_extra = torch.cat(to_cat).to(opt.device)
    total_samps = len(real_and_extra)
    opt.nzx = real.shape[2]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real.shape[3]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzz = real.shape[4]#+(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    m_noise3D = nn.ConstantPad3d(int(pad_noise), 0) if not opt.planar_convs else nn.ConstantPad3d(functions.create_planar_pad(pad_noise, opt.planar_convs), 0)
    m_image3D = nn.ConstantPad3d(int(pad_image), 0) if not opt.planar_convs else nn.ConstantPad3d(functions.create_planar_pad(pad_image, opt.planar_convs), 0)
    # Implementing depthless convolutions:
    # 1. m_noise3D = nn.ConstantPad3d((0, 0, int(pad_noise), int(pad_noise), int(pad_noise), int(pad_noise)), 0)
    # 2. m_image3D = nn.ConstantPad3d((0, 0, int(pad_noise), int(pad_noise), int(pad_noise), int(pad_noise)), 0)
    # 3. Change convs to use (ker_size, ker_size, 1) convolutions
    pad_discrim = int(((opt.ker_size - 1) * opt.num_layer_d) / 2)
    m_critic = nn.ConstantPad3d(pad_discrim, 0) if not opt.planar_convs else nn.ConstantPad3d(functions.create_planar_pad(pad_discrim, opt.planar_convs), 0)
    print(m_noise3D, m_image3D, m_critic)

    alpha = opt.alpha

    fixed_noise3D = functions.generate_noise3D([opt.nc_z,opt.nzx,opt.nzy,opt.nzz],device=opt.device,num_samp=total_samps)
    z_opt3D = torch.full(fixed_noise3D.shape, 0, device=opt.device)
    z_opt3D = m_noise3D(z_opt3D)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # 2: Trying LR warmup
    def lr(epoch):
        # 0: 0.1^5, 1: 0.1^4, 2: 0.1^3, 3: 0.1^2, 4: 0.1^1, 5: 0.1^0, 
        # 6: 0.99857^1, ...
        if epoch <= 5:
            lr_scale = 0.1**(5-epoch)
        else:
            # Calculated so that at epoch 1600, we are multiplying lr by 0.1 (opt.gamma)
            lr_scale = 0.99857**(epoch-5)
            # lr_scale = 1-((epoch - 5) / 2000)
        return lr_scale
        
    # schedulerD = optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lr)
    # schedulerG = optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lr)
    # 2: END
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    # schedulerD = torch.optim.lr_scheduler.CyclicLR(optimizerD, base_lr=0.01*opt.lr_d, max_lr=opt.lr_d, step_size_up=100, cycle_momentum=False)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []


    adversarial_loss = torch.nn.BCEWithLogitsLoss()
    ssim_target = TargetSimLoss(0.8, ssim).cuda()
    sim_loss = None
    assert opt.sim_type in ["vgg", "ssim", "ssim_target"]
    assert opt.sim_type != "vgg" # vgg not implemented for 3d
    assert opt.sim_boundary_type in ["start", "end"]
    if opt.sim_type == "ssim":
        sim_loss = SimLoss(use_harmonic=opt.harmonic_ssim).cuda()
    elif opt.sim_type == "ssim_target":
        sim_loss = ssim_target

    epoch = 0

    if opt.train_last_layer_longer and len(Gs) == opt.stop_scale:
        print("Training last layer for longer")
        opt.niter *= 2

    while epoch < int(opt.niter):
        if (Gs == []) & (opt.mode != 'SR_train'):
            z_opt3D = functions.generate_noise3D([1,opt.nzx,opt.nzy,opt.nzz], device=opt.device, num_samp=total_samps)
            z_opt3D = m_noise3D(z_opt3D.expand(total_samps,opt.nc_z,opt.nzx,opt.nzy,opt.nzz))
            noise_3D = functions.generate_noise3D([1,opt.nzx,opt.nzy,opt.nzz], device=opt.device)
            noise_3D = m_noise3D(noise_3D.expand(1,opt.nc_z,opt.nzx,opt.nzy,opt.nzz))
        else:
            noise_3D = functions.generate_noise3D([opt.nc_z,opt.nzx,opt.nzy,opt.nzz], device=opt.device)
            noise_3D = m_noise3D(noise_3D)

        SELECTED_IDX = random.choice(range(total_samps))
        SELECTED_REAL = real_and_extra[SELECTED_IDX][None]

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            valid = torch.full((1,1), 1.0).to(opt.device)
            fake_label = torch.full((1,1), 0.0).to(opt.device)

            # train with real
            netD.zero_grad()

            if not opt.sim_cond_d:
                input_d_real = SELECTED_REAL
                if opt.discrim_no_fewgan:
                    # Only show the original real image to the discriminator
                    input_d_real = real_and_extra[0][None]
            else:   
                ssim_channel = torch.ones(SELECTED_REAL.shape).to(opt.device)
                input_d_real = torch.cat([SELECTED_REAL, ssim_channel], axis=1)

            if opt.generate_with_critic:
                output_real = netD(m_critic(input_d_real)).to(opt.device)
                assert output_real.shape[2:] == input_d_real.shape[2:] and output_real.shape[0] == input_d_real.shape[0]
            else:
                output_real = netD(input_d_real).to(opt.device)
            #D_real_map = output.detach()
            if not opt.relativistic:
                errD_real = -output_real.mean()#-a
                errD_real.backward(retain_graph=True)


            # train with fake
            if (j==0) & (epoch == 0):
                if (Gs == []) & (opt.mode != 'SR_train'):
                    prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy,opt.nzz], 0, device=opt.device)
                    in_s = prev
                    prev = m_image3D(prev)
                    z_prev3D = torch.full([total_samps,opt.nc_z,opt.nzx,opt.nzy,opt.nzz], 0, device=opt.device)
                    in_s_z_opt = z_prev3D
                    z_prev3D = m_noise3D(z_prev3D)
                    opt.noise_amp = 1
                else:
                    prev = draw_concat3D(Gs,Ds,Zs,reals3D,NoiseAmp,in_s,'rand',m_noise3D,m_image3D,opt)
                    prev = m_image3D(prev)
                    assert in_s_z_opt.shape[:2] == real_and_extra.shape[:2], f"{in_s_z_opt.shape} versus {real_and_extra.shape}"
                    z_prev3D = draw_concat3D(Gs,Ds,Zs,reals3D,NoiseAmp,in_s_z_opt,'rec',m_noise3D,m_image3D,opt)
                    criterion = nn.MSELoss()
                    assert z_prev3D.shape[:2] == real_and_extra.shape[:2], f"{z_prev3D.shape} versus {real_and_extra.shape}"
                    RMSE = torch.sqrt(criterion(real_and_extra, z_prev3D))
                    opt.noise_amp = opt.noise_amp_init*RMSE
                    z_prev3D = m_image3D(z_prev3D)
            else:
                prev = draw_concat3D(Gs,Ds,Zs,reals3D,NoiseAmp,in_s,'rand',m_noise3D,m_image3D,opt)
                prev = m_image3D(prev)

            if opt.mode == 'paint_train':
                assert False, "not implemented"
                prev = functions.quant2centers(prev,centers)
                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise3D = noise_3D
            else:
                noise3D = opt.noise_amp*noise_3D+prev

            if len(Gs) > 0 and opt.generate_with_critic:
                if opt.detach_critic:
                    prev_discrim = netD(m_critic(prev.detach())).detach()
                else:
                    prev_discrim = netD(m_critic(prev.detach()))
                assert prev_discrim.shape == noise3D.shape
                fake = netG(torch.cat((noise3D.detach(), prev_discrim), dim=1),prev)
            else:
                fake = netG(noise3D.detach(),prev)

            if not opt.sim_cond_d:
                input_d_fake = fake
            else:
                ssim_channel = torch.full(fake.shape, ssim((fake.detach() + 1) / 2, (SELECTED_REAL + 1) / 2)).to(opt.device)
                input_d_fake = torch.cat([fake, ssim_channel], axis=1)

            if opt.generate_with_critic:
                output_fake = netD(m_critic(input_d_fake.detach()))
            else:
                output_fake = netD(input_d_fake.detach())
            if not opt.relativistic:
                errD_fake = output_fake.mean()
                errD_fake.backward(retain_graph=True)
            else:
                errD_real = adversarial_loss((output_real.mean() - output_fake.mean()).unsqueeze(0).unsqueeze(1), valid)
                errD_real.backward(retain_graph=True)
                errD_fake = adversarial_loss((output_fake.mean() - output_real.mean()).unsqueeze(0).unsqueeze(1), fake_label)
                errD_fake.backward(retain_graph=True)

            if opt.generate_with_critic:
                gradient_penalty = functions.calc_gradient_penalty(netD, m_critic(input_d_real), m_critic(input_d_fake), opt.lambda_grad, opt.device)
            else:
                gradient_penalty = functions.calc_gradient_penalty(netD, input_d_real, input_d_fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

        errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            netG.zero_grad()
            if opt.generate_with_critic:
                output = netD(m_critic(input_d_fake))
            else:
                output = netD(input_d_fake)
            #D_fake_map = output.detach()
            if not opt.relativistic:
                errG = -output.mean()
            else:
                errG = adversarial_loss((output_real.mean() - output.mean()).unsqueeze(0).unsqueeze(1), fake_label)
                errG += adversarial_loss((output.mean() - output_real.mean()).unsqueeze(0).unsqueeze(1), valid)
            errG.backward(retain_graph=True)
            
            # Similarity loss (only apply for sim alpha != 0)
            # if opt.linear_sim:
            #     fake_adjusted = (fake + 1) / 2
            #     real_adjusted = (SELECTED_REAL + 1) / 2
            #     assert fake_adjusted.shape == real_adjusted.shape
            #     ssim_loss = sim_loss(fake_adjusted, real_adjusted)
            #     if opt.linear_sim == 1: # decreasing linear
            #         alpha = (opt.stop_scale - len(Gs) + 1) * opt.sim_alpha
            #     else: # increasing linear
            #         alpha = (len(Gs) + 1) * opt.sim_alpha
            #     errG += alpha * ssim_loss
            # elif [...]
            if opt.sim_alpha != 0 and opt.sim_boundary_type == "start":
                if len(Gs) >= opt.sim_boundary:
                    fake_adjusted = (fake + 1) / 2
                    real_adjusted = (SELECTED_REAL + 1) / 2
                    assert fake_adjusted.shape == real_adjusted.shape
                    if opt.harmonic_ssim:
                        ssim_results = sim_loss(fake_adjusted.expand((3,)+fake_adjusted.shape[1:]), (real_and_extra + 1) / 2)
                        ssim_loss = harmonic_mean(ssim_results)
                    else:
                        ssim_loss = sim_loss(fake_adjusted, real_adjusted)
                    ssim_loss = opt.sim_alpha * ssim_loss
                    ssim_loss.backward(retain_graph=True)
            elif opt.sim_alpha != 0 and opt.sim_boundary_type == "end":
                if len(Gs) <= opt.sim_boundary:
                    fake_adjusted = (fake + 1) / 2
                    real_adjusted = (SELECTED_REAL + 1) / 2
                    assert fake_adjusted.shape == real_adjusted.shape
                    if opt.harmonic_ssim:
                        ssim_results = sim_loss(fake_adjusted.expand((3,)+fake_adjusted.shape[1:]), (real_and_extra + 1) / 2)
                        ssim_loss = harmonic_mean(ssim_results)
                    else:
                        ssim_loss = sim_loss(fake_adjusted, real_adjusted)
                    ssim_loss = opt.sim_alpha * ssim_loss
                    ssim_loss.backward(retain_graph=True)
            elif opt.sim_alpha != 0:
                assert False, "Incorrect use of sim alpha."

            if alpha!=0:
                loss = nn.L1Loss()
                assert z_opt3D.shape == z_prev3D.shape
                Z_opt = opt.noise_amp*z_opt3D+z_prev3D
                assert Z_opt.shape[:2] == real_and_extra.shape[:2], f"{Z_opt.shape} versus {real_and_extra.shape}"
                assert z_prev3D.shape[:2] == real_and_extra.shape[:2], f"{z_prev3D.shape} versus {real_and_extra.shape}"
                if len(Gs) > 0 and opt.generate_with_critic:
                    if opt.detach_critic:
                        prev_discrim = netD(m_critic(z_prev3D.detach())).detach()
                    else:
                        prev_discrim = netD(m_critic(z_prev3D.detach()))
                    assert prev_discrim.shape == Z_opt.shape
                    rec_loss = alpha*loss(netG(torch.cat((Z_opt.detach(), prev_discrim), dim=1),z_prev3D), real_and_extra)
                else:
                    rec_loss = alpha*loss(netG(Z_opt.detach(),z_prev3D), real_and_extra)
                # rec_loss = alpha*loss(netG(Z_opt[SELECTED_IDX][None].detach(),z_prev3D[SELECTED_IDX][None]), SELECTED_REAL)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt3D
                rec_loss = 0

            optimizerG.step()

        errG2plot.append(errG.detach()+rec_loss)
        z_opt2plot.append(rec_loss)

        if epoch % 25 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            # 3: UPDATED image saving (No more updates past 5/29)
            plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np3D(fake.detach(), opt=opt), vmin=0, vmax=1)
            if len(Gs) > 0 and opt.generate_with_critic:
                if opt.detach_critic:
                    prev_discrim = netD(m_critic(z_prev3D.detach())).detach()
                else:
                    prev_discrim = netD(m_critic(z_prev3D.detach()))
                assert prev_discrim.shape == z_prev3D.shape
                opt_imgs = netG(torch.cat((Z_opt.detach(), prev_discrim), dim=1), z_prev3D).detach()
            else:
                opt_imgs = netG(Z_opt.detach(), z_prev3D).detach()
            for idx, opt_img in enumerate(opt_imgs):
                plt.imsave('%s/z_prev_%d.png'    % (opt.outf, idx),  functions.convert_image_np3D(z_prev3D[idx][None], opt=opt), vmin=0, vmax=1)
                plt.imsave('%s/G(z_opt)_%d.png'    % (opt.outf, idx),  functions.convert_image_np3D(opt_img[None], opt=opt), vmin=0, vmax=1)
                plt.imsave('%s/real_%d.png'    % (opt.outf, idx),  functions.convert_image_np3D(real_and_extra[idx][None], opt=opt), vmin=0, vmax=1)
                for name, img in [("z_prev", z_prev3D[idx][None]), ("opt_img", opt_img[None]), ("real", real_and_extra[idx][None])]:
                    to_save = functions.convert_image_np3D(img, eval=True, opt=opt)
                    img = nib.Nifti1Image(to_save[:,:,:,0], np.eye(4))
                    nib.save(img, os.path.join(opt.outf, f"{name}_{idx}.nii.gz"))
            # 3: end
            #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


            torch.save(z_opt3D, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()

        epoch += 1

    functions.save_networks(netG,netD,z_opt3D,opt)
    return z_opt3D,in_s,in_s_z_opt,netG,netD   

def draw_concat3D(Gs,Ds,Zs,reals3D,NoiseAmp,in_s,mode,m_noise3D,m_image3D,opt):
    G_z = in_s
    pad_discrim = int(((opt.ker_size - 1) * opt.num_layer_d) / 2)
    assert not opt.planar_convs, "do not use"
    m_critic = nn.ConstantPad3d(pad_discrim, 0) if not opt.planar_convs else nn.ConstantPad3d(functions.create_planar_pad(pad_discrim, opt.planar_convs), 0)
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,D,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Ds,Zs,reals3D,reals3D[1:],NoiseAmp):
                if count == 0:
                    noise_shape = [1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise, Z_opt.shape[4] - 2 * pad_noise]
                    if opt.planar_convs:
                        noise_shape[opt.planar_convs] = Z_opt.shape[opt.planar_convs + 1]
                    z3D = functions.generate_noise3D(noise_shape, device=opt.device)
                    z3D = z3D.expand(1, opt.nc_z, z3D.shape[2], z3D.shape[3], z3D.shape[4])
                else:
                    noise_shape = [opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise, Z_opt.shape[4] - 2 * pad_noise]
                    if opt.planar_convs:
                        noise_shape[opt.planar_convs] = Z_opt.shape[opt.planar_convs + 1]
                    z3D = functions.generate_noise3D(noise_shape, device=opt.device)
                z3D = m_noise3D(z3D)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3],0:real_curr.shape[4]]
                G_z = m_image3D(G_z)
                z_in = noise_amp*z3D+G_z
                if count > 0 and opt.generate_with_critic:
                    if opt.detach_critic:
                        prev_discrim = D(m_critic(G_z.detach())).detach()
                    else:
                        prev_discrim = D(m_critic(G_z.detach()))
                    assert prev_discrim.shape == z_in.shape
                    G_z = G(torch.cat((z_in.detach(), prev_discrim), dim=1),G_z)
                else:
                    G_z = G(z_in.detach(),G_z)
                G_z = imresize3D(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3],0:real_next.shape[4]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,D,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Ds,Zs,reals3D,reals3D[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3], 0:real_curr.shape[4]]
                G_z = m_image3D(G_z)
                z_in = noise_amp*Z_opt+G_z
                if count > 0 and opt.generate_with_critic:
                    if opt.detach_critic:
                        prev_discrim = D(m_critic(G_z.detach())).detach()
                    else:
                        prev_discrim = D(m_critic(G_z.detach()))
                    assert prev_discrim.shape == z_in.shape
                    G_z = G(torch.cat((z_in.detach(), prev_discrim), dim=1),G_z)
                else:
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


def init_models(opt, scale_num):

    #generator initialization:
    if opt.generate_with_critic:
        netG = models.GeneratorConcatSkip2CleanAdd(opt, accepting_discrim_output=(scale_num > 0)).to(opt.device)
    else:
        netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG
