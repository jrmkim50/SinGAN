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
import SinGAN.vit_v_net as vitV

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
            if (not opt.vitV or opt.min_size == real_.shape[2]) and opt.pretrainG != 2:
                print("loading gen")
                G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            print("loading discrim")
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)), strict=False)

        z_curr,in_s,in_s_z_opt,G_curr,D_curr = train_single_scale3D(D_curr,G_curr,reals,extra_pyramids,Gs,Zs,in_s,in_s_z_opt,NoiseAmp,opt)

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

def make_slice(im, vgg_axis, mid_slice):
    # im: b,c,w,h,d
    im_slice_1, im_slice_2, im_slice_3 = None, None, None
    if vgg_axis == 1:
        im_slice_1 = im[:,:,mid_slice-1]
        im_slice_2 = im[:,:,mid_slice]
        im_slice_3 = im[:,:,mid_slice+1]
    elif vgg_axis == 2:
        im_slice_1 = im[:,:,:,mid_slice-1]
        im_slice_2 = im[:,:,:,mid_slice]
        im_slice_3 = im[:,:,:,mid_slice+1]
    else:
        im_slice_1 = im[:,:,:,:,mid_slice-1]
        im_slice_2 = im[:,:,:,:,mid_slice]
        im_slice_3 = im[:,:,:,:,mid_slice+1]
    im_slice = torch.cat([im_slice_1,im_slice_2,im_slice_3],1).float()
    return im_slice

class VGGLossWraper(nn.Module):
    def __init__(self, vgg_axis, scale_num):
        # scale_num == len(Gs)
        super(VGGLossWraper, self).__init__()
        self.loss = VGGLoss().cuda()
        assert vgg_axis in [1,2,3]
        self.vgg_axis = vgg_axis
        self.scale_num = scale_num
        self.scale_to_slice = {
            3: {
                1: 21, 2: 14, 3: 14
            },
            4: {
                1: 26, 2: 16, 3: 17
            },
            5: {
                1: 32, 2: 20, 3: 21
            },
            6: {
                1: 39, 2: 25, 3: 26
            }
        }

    def forward(self, fake, real):
        # fake and real are in 0-1 range when forward() called
        fake_slice = make_slice(fake, self.vgg_axis, self.scale_to_slice[self.scale_num][self.vgg_axis])
        real_slice = make_slice(real, self.vgg_axis, self.scale_to_slice[self.scale_num][self.vgg_axis])
        return self.loss(fake_slice, real_slice)

def harmonic_mean(nums):
    assert len(nums) > 0
    return len(nums) / torch.reciprocal(nums + 1e-16).sum()

def fillWithReal(real):
    realRef = torch.zeros_like(real)
    rectSize = (real.shape[2] // 2, real.shape[3] // 2, real.shape[4] // 2)
    rectStart = [random.randint(0, size) for size in rectSize]
    realRef[:,:,rectStart[0]:rectStart[0]+rectSize[0],
            rectStart[1]:rectStart[1]+rectSize[1],
            rectStart[2]:rectStart[2]+rectSize[2]] = real[:,:,rectStart[0]:rectStart[0]+rectSize[0],
            rectStart[1]:rectStart[1]+rectSize[1],
            rectStart[2]:rectStart[2]+rectSize[2]]
    return realRef

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
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2) if (not opt.unetG and not opt.vitV and opt.padd_size == 0) else 0
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2) if (not opt.unetG and not opt.vitV and opt.padd_size == 0) else 0
    m_noise3D = nn.ConstantPad3d(int(pad_noise), 0)
    m_image3D = nn.ConstantPad3d(int(pad_image), 0)

    alpha = opt.alpha

    fixed_noise3D = functions.generate_noise3D([opt.nc_z,opt.nzx,opt.nzy,opt.nzz],device=opt.device,num_samp=total_samps)
    z_opt3D = torch.full(fixed_noise3D.shape, 0, device=opt.device)
    z_opt3D = m_noise3D(z_opt3D)

    dParamsToUpdate = netD.parameters()
    if opt.pretrainD:
        dParamsToUpdate = []
        for name, param in netD.named_parameters():
            if param.requires_grad:
                dParamsToUpdate.append(param)
                print(name)

    # setup optimizer
    optimizerD = optim.Adam(dParamsToUpdate, lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # 2: Trying LR warmup
    def lr(epoch):
        # 0: 0.1^5, 1: 0.1^4, 2: 0.1^3, 3: 0.1^2, 4: 0.1^1, 5: 0.1^0, 
        # 6: decayLR^1, ...
        decayLR = opt.gamma**(1/(0.8*opt.niter))
        if epoch <= 5:
            lr_scale = 0.1**(5-epoch)
        else:
            # Calculated so that at epoch 1600, we are multiplying lr by 0.1 (opt.gamma)
            lr_scale = decayLR**(epoch-5)
            # lr_scale = 1-((epoch - 5) / 2000)
        return lr_scale
        
    if opt.warmup_d:
        schedulerD = optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lr)
    else:
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[0.8*opt.niter],gamma=opt.gamma)
    
    if opt.warmup_g:
        schedulerG = optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lr)
    else:
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[0.8*opt.niter],gamma=opt.gamma)
    # schedulerD = torch.optim.lr_scheduler.CyclicLR(optimizerD, base_lr=0.01*opt.lr_d, max_lr=opt.lr_d, step_size_up=100, cycle_momentum=False)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    sim_loss = None
    assert opt.sim_type in ["vgg", "ssim", "medical_net"]
    assert opt.sim_boundary_type in ["start", "end"]
    if opt.sim_type == "ssim":
        sim_loss = SimLoss(use_harmonic=opt.harmonic_ssim).cuda()
    elif opt.sim_type == "vgg":
        sim_loss = VGGLossWraper(opt.vgg_axis, len(Gs))
    elif opt.sim_type == "medical_net":
        sim_loss = MedicalNetLoss(normalize=opt.normalize_medical_net, model=opt.medical_net_model)

    epoch = 0

    niter = opt.niter

    if opt.train_last_layer_longer and len(Gs) == opt.stop_scale:
        print("Training last layer for longer")
        niter *= 2
    elif opt.train_first_layers_longer and len(Gs) < opt.train_first_layers_longer:
        print("Training layer for longer")
        niter *= 2

    total_count = 0
    num_correct = 0

    if opt.pretrainDiscrim:
        for j in range(max(niter // 4, 1)):
            SELECTED_IDX = random.choice(range(total_samps))
            SELECTED_REAL = real_and_extra[SELECTED_IDX][None]
            netD.zero_grad()
            input_d_real = SELECTED_REAL
            output_real = netD(input_d_real).to(opt.device)
            errD_real = -output_real.mean()#-a
            errD_real.backward()
            optimizerD.step()

            if j % 25 == 0:
                print(f"pretrain step: {j}/{max(niter // 4, 1)}")


    if (Gs == []) & (opt.mode != 'SR_train'):
        prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy,opt.nzz], 0, device=opt.device)
        in_s = prev
        prev = m_image3D(prev)
        z_prev3D = torch.full([total_samps,opt.nc_z,opt.nzx,opt.nzy,opt.nzz], 0, device=opt.device)
        in_s_z_opt = z_prev3D
        z_prev3D = m_noise3D(z_prev3D)
        opt.noise_amp = 1
    else:
        prev = draw_concat3D(Gs,Zs,reals3D,NoiseAmp,in_s,'rand',m_noise3D,m_image3D,opt,extra_pyramids)
        prev = m_image3D(prev)
        assert in_s_z_opt.shape[:2] == real_and_extra.shape[:2], f"{in_s_z_opt.shape} versus {real_and_extra.shape}"
        z_prev3D = draw_concat3D(Gs,Zs,reals3D,NoiseAmp,in_s_z_opt,'rec',m_noise3D,m_image3D,opt,extra_pyramids)
        criterion = nn.MSELoss()
        assert z_prev3D.shape[:2] == real_and_extra.shape[:2], f"{z_prev3D.shape} versus {real_and_extra.shape}"
        RMSE = torch.sqrt(criterion(real_and_extra, z_prev3D))
        opt.noise_amp = opt.noise_amp_init*RMSE
        z_prev3D = m_image3D(z_prev3D)

    if opt.pretrainGen:
        for j in range(max(niter // 4, 1)):
            if (Gs == []) & (opt.mode != 'SR_train'):
                z_opt3D = functions.generate_noise3D([1,opt.nzx,opt.nzy,opt.nzz], device=opt.device, num_samp=total_samps)
                z_opt3D = m_noise3D(z_opt3D.expand(total_samps,opt.nc_z,opt.nzx,opt.nzy,opt.nzz))
            netG.zero_grad()
            loss = nn.L1Loss()
            Z_opt = opt.noise_amp*z_opt3D+z_prev3D
            assert Z_opt.shape[:2] == real_and_extra.shape[:2], f"{Z_opt.shape} versus {real_and_extra.shape}"
            assert z_prev3D.shape[:2] == real_and_extra.shape[:2], f"{z_prev3D.shape} versus {real_and_extra.shape}"
            if not opt.update_in_one_go:
                for idx in range(total_samps):
                    fake_recon = netG(Z_opt.detach()[idx][None],z_prev3D[idx][None])
                    rec_loss = (alpha / total_samps)*loss(fake_recon, real_and_extra[idx][None])
                    rec_loss.backward(retain_graph=True)
            else:
                fake_recon = netG(Z_opt.detach(),z_prev3D)
                rec_loss = alpha*loss(fake_recon, real_and_extra)
                rec_loss.backward(retain_graph=True)
            optimizerG.step()

    while epoch < int(niter):
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

            SELECTED_IDX = random.choice(range(total_samps))
            SELECTED_REAL = real_and_extra[SELECTED_IDX][None]

            # train with fake
            if not ((j == 0) & (epoch == 0)):
                prev = draw_concat3D(Gs,Zs,reals3D,NoiseAmp,in_s,'rand',m_noise3D,m_image3D,opt,extra_pyramids)
                prev = m_image3D(prev)

            input_d_real = SELECTED_REAL

            instanceNoiseVariance = 0.1 - 0.1*epoch/niter # anneal from 0.1 to 0
            realVariance = functions.generate_noise_with_variance(input_d_real.shape, opt.device, instanceNoiseVariance) if opt.noisyDiscrim else 0

            output_real = netD(input_d_real + realVariance).to(opt.device)
            errD_real = -output_real.mean()#-a
            if not opt.update_in_one_go:
                errD_real.backward()

            if output_real.detach().mean() > 0:
                num_correct += 1
            total_count += 1

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise3D = noise_3D
            else:
                noise3D = opt.noise_amp*noise_3D+prev

            if opt.combineWithR:
                noise3D = torch.cat([noise3D, m_noise3D(fillWithReal(SELECTED_REAL))], dim=1)

            fake = netG(noise3D.detach(),prev)

            assert fake.shape == input_d_real.shape

            input_d_fake = fake
            fakeVariance = functions.generate_noise_with_variance(input_d_fake.shape, opt.device, instanceNoiseVariance) if opt.noisyDiscrim else 0

            output_fake = netD(input_d_fake.detach() + fakeVariance)
            errD_fake = output_fake.mean()
            if not opt.update_in_one_go:
                errD_fake.backward()

            if output_fake.detach().mean() < 0:
                num_correct += 1
            total_count += 1

            gradient_penalty = functions.calc_gradient_penalty(netD, input_d_real, input_d_fake, opt.lambda_grad, opt.device)
            if not opt.update_in_one_go:
                gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            if opt.update_in_one_go:
                errD.backward()
            optimizerD.step()

        errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
        for j in range(opt.Gsteps):
            netG.zero_grad()

            if opt.remakeFake:
                fake = netG(noise3D.detach(),prev)
                input_d_fake = fake

            SELECTED_IDX = random.choice(range(total_samps))
            SELECTED_REAL = real_and_extra[SELECTED_IDX][None]
            
            if not opt.reconLossOnly:
                output = netD(input_d_fake)
                #D_fake_map = output.detach()
                errG = -output.mean()
                
                if not opt.update_in_one_go:
                    errG.backward(retain_graph=True)
                
                should_compute_sim = (opt.sim_alpha != 0 and 
                                    ((opt.sim_boundary_type == "start" and len(Gs) >= opt.sim_boundary) or 
                                    (opt.sim_boundary_type == "end" and len(Gs) <= opt.sim_boundary)))

                ssim_loss = 0

                if should_compute_sim:
                    fake_adjusted = (fake + 1) / 2
                    real_adjusted = (SELECTED_REAL + 1) / 2
                    if opt.sim_loss_one_image:
                        real_adjusted = (real_and_extra[0][None] + 1) / 2
                    assert fake_adjusted.shape == real_adjusted.shape
                    if opt.harmonic_ssim:
                        ssim_results = sim_loss(fake_adjusted.expand((total_samps,)+fake_adjusted.shape[1:]), (real_and_extra + 1) / 2)
                        ssim_loss = harmonic_mean(ssim_results)
                    else:
                        ssim_loss = sim_loss(fake_adjusted, real_adjusted)
                    ssim_loss = opt.sim_alpha * ssim_loss
                    if not opt.update_in_one_go:
                        ssim_loss.backward(retain_graph=True)
            else:
                errG = 0
                ssim_loss = 0

            rec_loss = 0

            if alpha!=0:
                loss = nn.L1Loss()
                assert z_opt3D.shape == z_prev3D.shape
                Z_opt = opt.noise_amp*z_opt3D+z_prev3D
                assert Z_opt.shape[:2] == real_and_extra.shape[:2], f"{Z_opt.shape} versus {real_and_extra.shape}"
                assert z_prev3D.shape[:2] == real_and_extra.shape[:2], f"{z_prev3D.shape} versus {real_and_extra.shape}"
                if not opt.update_in_one_go:
                    for idx in range(total_samps):
                        zNoise = torch.cat([Z_opt.detach()[idx][None], m_noise3D(fillWithReal(SELECTED_REAL))], dim=1) if opt.combineWithR else Z_opt.detach()[idx][None]
                        fake_recon = netG(zNoise,z_prev3D[idx][None])
                        rec_loss = (alpha / total_samps)*loss(fake_recon, real_and_extra[idx][None])
                        rec_loss.backward(retain_graph=True)
                else:
                    zNoise = torch.cat([Z_opt.detach(), m_noise3D(fillWithReal(real_and_extra))], dim=1) if opt.combineWithR else Z_opt.detach()
                    fake_recon = netG(zNoise,z_prev3D)
                    rec_loss = alpha*loss(fake_recon, real_and_extra)
            else:
                Z_opt = z_opt3D
                rec_loss = 0

            if opt.update_in_one_go:
                errGTotal = errG + ssim_loss + rec_loss
                errGTotal.backward(retain_graph=True)

            optimizerG.step()

            for _ in range(opt.extraRecon):
                assert opt.extraRecon > 0, "opt.extraRecon issue"
                loss = nn.L1Loss()
                assert z_opt3D.shape == z_prev3D.shape
                Z_opt = opt.noise_amp*z_opt3D+z_prev3D
                assert Z_opt.shape[:2] == real_and_extra.shape[:2], f"{Z_opt.shape} versus {real_and_extra.shape}"
                assert z_prev3D.shape[:2] == real_and_extra.shape[:2], f"{z_prev3D.shape} versus {real_and_extra.shape}"
                if not opt.update_in_one_go:
                    for idx in range(total_samps):
                        fake_recon = netG(Z_opt.detach()[idx][None],z_prev3D[idx][None])
                        rec_loss = (alpha / total_samps)*loss(fake_recon, real_and_extra[idx][None])
                        rec_loss.backward(retain_graph=True)
                else:
                    fake_recon = netG(Z_opt.detach(),z_prev3D)
                    rec_loss = alpha*loss(fake_recon, real_and_extra)
                    rec_loss.backward(retain_graph=True)
                optimizerG.step()

        if not opt.reconLossOnly:
            errG2plot.append(errG.detach()+rec_loss.detach())
            z_opt2plot.append(rec_loss.detach())

        if epoch % 25 == 0 or epoch == (niter-1):
            print('scale %d:[%d/%d]; d_real_fake: [%.3f] [%.3f]; d_err [%.3f]; errG [%.3f] [%.3f]' % (len(Gs), epoch, niter, output_real.detach().mean(), output_fake.detach().mean(), errD2plot[-1], errG.detach().item(), rec_loss.detach().item()))

        if epoch % 500 == 0 or epoch == (niter-1):
            # 3: UPDATED image saving (No more updates past 5/29)
            plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np3D(fake.detach(), opt=opt), vmin=0, vmax=1)
            zNoise = torch.cat([Z_opt.detach(), m_noise3D(fillWithReal(real_and_extra))], dim=1) if opt.combineWithR else Z_opt.detach()
            opt_imgs = netG(zNoise, z_prev3D).detach()
            for idx, opt_img in enumerate(opt_imgs):
                plt.imsave('%s/z_prev_%d.png'    % (opt.outf, idx),  functions.convert_image_np3D(z_prev3D[idx][None], opt=opt), vmin=0, vmax=1)
                plt.imsave('%s/G(z_opt)_%d.png'    % (opt.outf, idx),  functions.convert_image_np3D(opt_img[None], opt=opt), vmin=0, vmax=1)
                plt.imsave('%s/real_%d.png'    % (opt.outf, idx),  functions.convert_image_np3D(real_and_extra[idx][None], opt=opt), vmin=0, vmax=1)
                for name, img in [("z_prev", z_prev3D[idx][None]), ("opt_img", opt_img[None]), ("real", real_and_extra[idx][None])]:
                    to_save = functions.convert_image_np3D(img, eval=True, opt=opt)
                    img = nib.Nifti1Image(to_save[:,:,:,:], np.eye(4))
                    nib.save(img, os.path.join(opt.outf, f"{name}_{idx}.nii.gz"))


            torch.save(z_opt3D, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()

        epoch += 1

    print(f"DISCRIMINATOR ACCURACY ({num_correct}/{total_count}):", num_correct/total_count)

    functions.save_networks(netG,netD,z_opt3D,opt)
    return z_opt3D,in_s,in_s_z_opt,netG,netD 

def draw_concat3D(Gs,Zs,reals3D,NoiseAmp,in_s,mode,m_noise3D,m_image3D,opt,extra_pyramids):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2) if (not opt.unetG and not opt.vitV and opt.padd_size == 0) else 0
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
                if opt.combineWithR:
                    randomIndex = random.choice(range(1+len(extra_pyramids)))
                    imagesForLevel = [reals3D[count],] + [pyramid[count] for pyramid in extra_pyramids]
                    z_in = torch.cat([z_in, m_noise3D(fillWithReal(imagesForLevel[randomIndex]))], dim=1)
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
                if opt.combineWithR:
                    imagesForLevel = torch.cat([reals3D[count],] + [pyramid[count] for pyramid in extra_pyramids])
                    z_in = torch.cat([z_in, m_noise3D(fillWithReal(imagesForLevel))], dim=1)
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
    if not opt.vitV or real_shape[2] == 32: # use regular model for first stage of vitV
        netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device) if (not opt.unetG and not opt.vitV) else models.Unet(opt, True).to(opt.device)
    else:
        netG = vitV.ViTVNet(vitV.get_3DReg_config(), real_shape[2:]).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    if opt.pretrainG == 1 and scale_num == 0:
        netG.load_state_dict(torch.load("TrainedModels/rat_two_channel_split.ni/scale_factor=0.850,num_layers=6,sim_alpha=0.300,sim_boundary=3,sim_boundary_type=start,use_attn_g=1,use_attn_end_g=0,use_attn_d=1,use_attn_end_d=0,nfc=32,min_size=20,few_gan=5,num_layer_d=4,split,warm_d,update_in_one_go,alpha=50,reconLoss,niter=3000/0/netG.pth", map_location=torch.device('cpu')))
    elif opt.pretrainG == 2:
        netG.load_state_dict(torch.load(f"TrainedModels/rat_two_channel_split.ni/scale_factor=0.850,num_layers=6,sim_alpha=0.300,sim_boundary=3,sim_boundary_type=start,use_attn_g=1,use_attn_end_g=0,use_attn_d=1,use_attn_end_d=0,nfc=32,min_size=20,few_gan=5,num_layer_d=4,split,warm_d,update_in_one_go,alpha=50,reconLoss,niter=3000/{scale_num}/netG.pth", map_location=torch.device('cpu')))
    print(netG)

    #discriminator initialization:
    if opt.pretrainD:
        netD = models.FinetuneNet(opt).to(opt.device)
        netD.apply(models.weights_init)
        netD.load_state_dict(torch.load('monai_wholebody_ct/models/model.pth'), strict=False)
        netD.convInit.requires_grad_(False) # freeze convInit
        netD.down_layers.requires_grad_(False) # freeze the pretrained layer
    else:
        netD = models.WDiscriminator(opt).to(opt.device) if not opt.unetD else models.Unet(opt, False).to(opt.device)
        netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG