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

def train(opt,Gs,Zs,reals,NoiseAmp):
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
        plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np3D(reals[scale_num]), vmin=0, vmax=1)

        D_curr,G_curr = init_models(opt)
        if (nfc_prev==opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))

        z_curr,in_s,in_s_z_opt,G_curr = train_single_scale3D(D_curr,G_curr,reals,extra_pyramids,Gs,Zs,in_s,in_s_z_opt,NoiseAmp,opt)

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
    return

class TargetSimLoss(nn.Module):
    def __init__(self, target, ssim):
        super(TargetSimLoss, self).__init__()
        self.target = target
        self.ssim = ssim

    def forward(self, fake, real):
        return torch.abs(self.ssim(fake, real) - self.target)
    
class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()

    def forward(self, fake, real):
        return -1 * ssim(fake, real)

def train_single_scale3D(netD,netG,reals3D,extra_pyramids,Gs,Zs,in_s,in_s_z_opt,NoiseAmp,opt,centers=None):
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
    m_noise3D = nn.ConstantPad3d(int(pad_noise), 0)
    m_image3D = nn.ConstantPad3d(int(pad_image), 0)

    alpha = opt.alpha

    fixed_noise3D = functions.generate_noise3D([opt.nc_z,opt.nzx,opt.nzy,opt.nzz],device=opt.device,num_samp=total_samps)
    z_opt3D = torch.full(fixed_noise3D.shape, 0, device=opt.device)
    z_opt3D = m_noise3D(z_opt3D)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []


    ssim_target = TargetSimLoss(0.8, ssim).cuda()
    sim_loss = None
    assert opt.sim_type in ["vgg", "ssim", "ssim_target"]
    assert opt.sim_type != "vgg" # vgg not implemented for 3d
    assert opt.sim_boundary_type in ["start", "end"]
    if opt.sim_type == "ssim":
        sim_loss = SimLoss().cuda()
    elif opt.sim_type == "vgg":
        sim_loss = vgg_loss
    elif opt.sim_type == "ssim_target":
        sim_loss = ssim_target

    for epoch in range(opt.niter):
        if (Gs == []) & (opt.mode != 'SR_train'):
            z_opt3D = functions.generate_noise3D([1,opt.nzx,opt.nzy,opt.nzz], device=opt.device, num_samp=total_samps)
            z_opt3D = m_noise3D(z_opt3D.expand(total_samps,opt.nc_z,opt.nzx,opt.nzy,opt.nzz))
            noise_3D = functions.generate_noise3D([1,opt.nzx,opt.nzy,opt.nzz], device=opt.device)
            noise_3D = m_noise3D(noise_3D.expand(1,opt.nc_z,opt.nzx,opt.nzy,opt.nzz))
        else:
            noise_3D = functions.generate_noise3D([opt.nc_z,opt.nzx,opt.nzy,opt.nzz], device=opt.device)
            noise_3D = m_noise3D(noise_3D)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            netD.zero_grad()

            output = netD(real_and_extra).to(opt.device)
            #D_real_map = output.detach()
            errD_real = -output.mean()#-a
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

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
                    prev = draw_concat3D(Gs,Zs,reals3D,NoiseAmp,in_s,'rand',m_noise3D,m_image3D,opt)
                    prev = m_image3D(prev)
                    assert in_s_z_opt.shape[:2] == real_and_extra.shape[:2], f"{in_s_z_opt.shape} versus {real_and_extra.shape}"
                    z_prev3D = draw_concat3D(Gs,Zs,reals3D,NoiseAmp,in_s_z_opt,'rec',m_noise3D,m_image3D,opt)
                    criterion = nn.MSELoss()
                    assert z_prev3D.shape[:2] == real_and_extra.shape[:2], f"{z_prev3D.shape} versus {real_and_extra.shape}"
                    RMSE = torch.sqrt(criterion(real_and_extra, z_prev3D))
                    opt.noise_amp = opt.noise_amp_init*RMSE
                    z_prev3D = m_image3D(z_prev3D)
            else:
                prev = draw_concat3D(Gs,Zs,reals3D,NoiseAmp,in_s,'rand',m_noise3D,m_image3D,opt)
                prev = m_image3D(prev)

            if opt.mode == 'paint_train':
                assert False, "not implemented"
                prev = functions.quant2centers(prev,centers)
                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise3D = noise_3D
            else:
                noise3D = opt.noise_amp*noise_3D+prev

            fake = netG(noise3D.detach(),prev)
            output = netD(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            if errD_fake > 0 and opt.sim_loss_d:
                # Similarity loss (only apply for sim alpha != 0)
                if opt.sim_alpha != 0 and opt.sim_boundary_type == "start":
                    if len(Gs) >= opt.sim_boundary:
                        fake_adjusted = (fake + 1) / 2
                        real_idx = random.choice(range(real_and_extra.shape[0]))
                        real_adjusted = (real_and_extra[real_idx][None] + 1) / 2
                        # If fake image is very realistic, sim_loss returns -1, and we should not increase
                        # errD_fake by too much. If fake image is unrealistic but discrim said image was real
                        # we should penalize more heavily.
                        errD_fake += opt.sim_alpha * (1+sim_loss(fake_adjusted, real_adjusted))
                elif opt.sim_alpha != 0 and opt.sim_boundary_type == "end":
                    if len(Gs) <= opt.sim_boundary:
                        fake_adjusted = (fake + 1) / 2
                        real_idx = random.choice(range(real_and_extra.shape[0]))
                        real_adjusted = (real_and_extra[real_idx][None] + 1) / 2
                        # If fake image is very realistic, sim_loss returns -1, and we should not increase
                        # errD_fake by too much. If fake image is unrealistic but discrim said image was real
                        # we should penalize more heavily.
                        errD_fake += opt.sim_alpha * (1+sim_loss(fake_adjusted, real_adjusted))

            gradient_penalty = functions.calc_gradient_penalty(netD, real_and_extra, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

        errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            netG.zero_grad()
            output = netD(fake)
            #D_fake_map = output.detach()
            errG = -output.mean()
            # Similarity loss (only apply for sim alpha != 0)
            if opt.sim_alpha != 0 and opt.sim_boundary_type == "start":
                if len(Gs) >= opt.sim_boundary:
                    fake_adjusted = (fake + 1) / 2
                    real_idx = random.choice(range(real_and_extra.shape[0]))
                    real_adjusted = (real_and_extra[real_idx][None] + 1) / 2
                    assert fake_adjusted.shape == real_adjusted.shape
                    errG += opt.sim_alpha * sim_loss(fake_adjusted, real_adjusted)
            elif opt.sim_alpha != 0 and opt.sim_boundary_type == "end":
                if len(Gs) <= opt.sim_boundary:
                    fake_adjusted = (fake + 1) / 2
                    real_idx = random.choice(range(real_and_extra.shape[0]))
                    real_adjusted = (real_and_extra[real_idx][None] + 1) / 2
                    assert fake_adjusted.shape == real_adjusted.shape
                    errG += opt.sim_alpha * sim_loss(fake_adjusted, real_adjusted)
            elif opt.sim_alpha != 0:
                assert False, "Incorrect use of sim alpha."
            errG.backward(retain_graph=True)

            if alpha!=0:
                loss = nn.L1Loss()
                assert z_opt3D.shape == z_prev3D.shape
                Z_opt = opt.noise_amp*z_opt3D+z_prev3D
                assert Z_opt.shape[:2] == real_and_extra.shape[:2], f"{Z_opt.shape} versus {real_and_extra.shape}"
                assert z_prev3D.shape[:2] == real_and_extra.shape[:2], f"{z_prev3D.shape} versus {real_and_extra.shape}"
                rec_loss = alpha*loss(netG(Z_opt.detach(),z_prev3D),real_and_extra)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt3D
                rec_loss = 0

            optimizerG.step()

        errG2plot.append(errG.detach()+rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)

        if epoch % 25 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np3D(fake.detach()), vmin=0, vmax=1)
            opt_imgs = netG(Z_opt.detach(), z_prev3D).detach()
            for idx, opt_img in enumerate(opt_imgs):
                plt.imsave('%s/G(z_opt)_%d.png'    % (opt.outf, idx),  functions.convert_image_np3D(opt_img[None]), vmin=0, vmax=1)
            #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


            torch.save(z_opt3D, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG,netD,z_opt3D,opt)
    return z_opt3D,in_s,in_s_z_opt,netG    

def draw_concat3D(Gs,Zs,reals3D,NoiseAmp,in_s,mode,m_noise3D,m_image3D,opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals3D,reals3D[1:],NoiseAmp):
                if count == 0:
                    z3D = functions.generate_noise3D([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise, Z_opt.shape[4] - 2 * pad_noise], device=opt.device)
                    z3D = z3D.expand(1, opt.nc_z, z3D.shape[2], z3D.shape[3], z3D.shape[4])
                else:
                    z3D = functions.generate_noise3D([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise, Z_opt.shape[4] - 2 * pad_noise], device=opt.device)
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


def init_models(opt):

    #generator initialization:
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
