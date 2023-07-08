import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
#from skimage import morphology
#from skimage import filters
from SinGAN.imresize import imresize, imresize3D
import os
import random
import nibabel as nib
from sklearn.cluster import KMeans


# custom weights initialization called on netG and netD

def read_image(opt):
    assert False, "should not be called in 3d mode"
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x

def read_image3D(opt):
    # Channel axis should come at the end!
    x = nib.load('%s/%s' % (opt.input_dir,opt.input_name)).get_fdata()
    if len(x.shape) == 3:
        # Add channel axis
        x = x[:,:,:,None]
    if opt.split_image:
        assert len(x.shape) == 4
        assert x.shape[2] == 128
        x = np.concatenate((x[:,:,:64], x[:,:,64:]), axis=3)
    opt.nc_im = x.shape[-1]
    opt.nc_z = x.shape[-1]
    x = np2torch3D(x,opt)
    extra_images = []
    for num in range(opt.few_gan):
        # FewGAN reference image is of the form [opt.input_dir]/[opt.input_name (chop off file extension)]_[num].[extension]
        file_name = opt.input_name.split(".")[0]
        extension = ".".join(opt.input_name.split(".")[1:]) # Extension has no leading period
        reference = nib.load(f"{opt.input_dir}/{file_name}_{num}.{extension}").get_fdata()
        if len(reference.shape) == 3:
            # Add channel axis
            reference = reference[:,:,:,None]
        if opt.split_image:
            assert len(reference.shape) == 4
            assert reference.shape[2] == 128
            reference = np.concatenate((reference[:,:,:64], reference[:,:,64:]), axis=3)
        reference = np2torch3D(reference,opt)
        extra_images.append(reference)
    return x, extra_images

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)

#def denorm2image(I1,I2):
#    out = (I1-I1.mean())/(I1.max()-I1.min())
#    out = out*(I2.max()-I2.min())+I2.mean()
#    return out#.clamp(I2.min(), I2.max())

#def norm2image(I1,I2):
#    out = (I1-I2.mean())*2
#    return out#.clamp(I2.min(), I2.max())

def convert_image_np(inp):
    assert False, "should not be called in 3d mode"
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp,0,1)
    return inp

def convert_image_np3D(inp,eval=False,opt=None):
    assert opt
    inp = denorm(inp)
    inp = move_to_cpu(inp[-1,:,:,:,:])
    inp = inp.numpy().transpose((1,2,3,0))
    # w,h,d,channel
    inp = np.clip(inp,0,1)
    if opt.split_image:
        n_channels = inp.shape[-1]
        assert n_channels % 2 == 0
        # We unfold on axis 2 because we initially split at axis 2
        inp = np.concatenate((inp[:,:,:,:n_channels // 2], inp[:,:,:,n_channels // 2:]), axis=2)
        assert inp.shape[-1] > 0 and len(inp.shape) == 4
    if eval:
        return inp
    return inp[:,inp.shape[1] // 2,:, 0]

def unfold_tensor(inp):
    n_channels = inp.shape[1]
    assert n_channels % 2 == 0
    # if we would fold a tensor of b,c,w,h,d, we would split on axis 4, so we unfold on axis 4
    return torch.cat((inp[:,:n_channels // 2], inp[:,n_channels // 2:]), dim=4)

def save_image(real_cpu,receptive_feild,ncs,epoch_num,file_name):
    fig,ax = plt.subplots(1)
    if ncs==1:
        ax.imshow(real_cpu.view(real_cpu.size(2),real_cpu.size(3)),cmap='gray')
    else:
        #ax.imshow(convert_image_np(real_cpu[0,:,:,:].cpu()))
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0,0),receptive_feild,receptive_feild,linewidth=5,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)

def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
    # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
    # inp = std*
    return inp

def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    assert False, "should not be called in 3d mode"
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise

def generate_noise3D(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), round(size[3]/scale), device=device)
        noise = upsampling3D(noise,size[1], size[2],size[3])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], size[3], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], size[3], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], size[3], device=device)
    return noise


def plot_learning_curves(G_loss,D_loss,epochs,label1,label2,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,G_loss,n,D_loss)
    #plt.title('loss')
    #plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend([label1,label2],loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def plot_learning_curve(loss,epochs,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def upsampling3D(im,sx,sy,sz):
    m = nn.Upsample(size=[round(sx),round(sy),round(sz)],mode='trilinear',align_corners=True)
    return m(im)

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    # fake_data = fake_data.expand(real_data.size())
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def read_image_dir(dir,opt):
    assert False, "not used in 3d mode"
    x = img.imread('%s' % dir)
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x

def np2torch(x,opt):
    assert False, "do not call in 3d mode"
    if opt.nc_im >= 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor)
    x = norm(x)
    return x

def np2torch3D(x,opt):
    # x starts as w,h,d,channels
    x = x[:,:,:,:,None]
    # x is now batch,channels,w,h,d and is in 0-1 range
    x = x.transpose((4, 3, 0, 1, 2))
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor)
    x = norm(x)
    return x

def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = x[:, :, 0:3]
    return x

def save_networks(netG,netD,z,opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))

def adjust_scales2image(real_,opt):
    # real_: [batch_size, channels, width, height] (old)
    # real_: [batch_size, channels, width, height, depth] (new)
    #opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    opt.num_scales = math.ceil(
        (math.log(
            math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3], real_.shape[4])), 1), 
            opt.scale_factor_init))) + 1
    scale2stop = math.ceil(
        math.log(
            min([
                opt.max_size, 
                max([real_.shape[2], real_.shape[3], real_.shape[4]])
            ]) / max([real_.shape[2], real_.shape[3], real_.shape[4]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3], real_.shape[4]]),1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize3D(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(
        opt.min_size/(min(real.shape[2],real.shape[3], real_.shape[4])),
        1/(opt.stop_scale))
    scale2stop = math.ceil(
        math.log(
            min([
                opt.max_size, 
                max([real_.shape[2], real_.shape[3], real_.shape[4]])
            ]) / max([real_.shape[2], real_.shape[3], real_.shape[4]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def adjust_scales2image_SR(real_,opt):
    opt.min_size = 18
    opt.num_scales = int((math.log(opt.min_size / min(real_.shape[2], real_.shape[3]), opt.scale_factor_init))) + 1
    scale2stop = int(math.log(min(opt.max_size , max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = int(math.log(min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def creat_reals_pyramid(real,reals,opt):
    real = real[:,0:3,:,:]
    for i in range(0,opt.stop_scale+1,1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        curr_real = imresize(real,scale,opt)
        reals.append(curr_real)
    return reals

def creat_reals_pyramid3D(real,reals,opt):
    real = real[:,:,:,:]
    for i in range(0,opt.stop_scale+1,1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        curr_real = imresize3D(real,scale,opt)
        reals.append(curr_real)
    return reals


def load_trained_pyramid(opt, mode_='train'):
    #dir = 'TrainedModels/%s/scale_factor=%f' % (opt.input_name[:-4], opt.scale_factor_init)
    mode = opt.mode
    opt.mode = 'train'
    if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
        opt.mode = mode
    dir = generate_dir2save(opt)
    if opt.train_dir:
        dir = 'TrainedModels/%s/%s' % (opt.input_name[:-4], opt.train_dir)
    if(os.path.exists(dir)):
        Gs = torch.load('%s/Gs.pth' % dir)
        Zs = torch.load('%s/Zs.pth' % dir)
        reals = torch.load('%s/reals.pth' % dir)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir)
    else:
        print('no appropriate trained model is exist, please train first')
    opt.mode = mode
    return Gs,Zs,reals,NoiseAmp

def generate_in2coarsest3D(reals,scale_v,scale_h,scale_z,opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling3D(real, scale_v * real.shape[2], scale_h * real.shape[3], scale_h * real.shape[4])
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=opt.device)
    else: #if n!=0
        in_s = upsampling3D(real_down, real_down.shape[2], real_down.shape[3], real_down.shape[4])
    return in_s

def generate_in2coarsest(reals,scale_v,scale_h,opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=opt.device)
    else: #if n!=0
        in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s

def create_planar_pad(pad_size, planar_axis):
    assert planar_axis in [1,2,3]
    if planar_axis == 1:
        # kernel size is (1,ker_size,ker_size)
        return (int(pad_size), int(pad_size), int(pad_size), int(pad_size), 0, 0)
    elif planar_axis == 2:
        # kernel size is (ker_size,1,ker_size)
        return (int(pad_size), int(pad_size), 0, 0, int(pad_size), int(pad_size))
    elif planar_axis == 3:
        return (0, 0, int(pad_size), int(pad_size), int(pad_size), int(pad_size))

def generate_dir2save(opt):
    dir2save = None
    if (opt.mode == 'train') | (opt.mode == 'SR_train'):
        dir2save = f'TrainedModels/{opt.input_name[:-4]}/scale_factor={opt.scale_factor_init:.3f},num_layers={opt.num_layer},sim_alpha={opt.sim_alpha:.3f},sim_boundary={opt.sim_boundary},sim_boundary_type={opt.sim_boundary_type},use_attn_g={opt.use_attention_g},use_attn_end_g={opt.use_attention_end_g},use_attn_d={opt.use_attention_d},use_attn_end_d={opt.use_attention_end_d},nfc={opt.nfc_init},min_size={opt.min_size},few_gan={opt.few_gan},sim_cond_d={opt.sim_cond_d},num_layer_d={opt.num_layer_d}{"_groupnorm" if opt.groupnorm else ""}{",prelu" if opt.prelu else ""}{",rel" if opt.relativistic else ""}{",criticGenerator" if opt.generate_with_critic else ""}{"_detach" if opt.detach_critic else ""}{",train_last_longer" if opt.train_last_layer_longer else ""}{",split" if opt.split_image else ""}{",min_ssim" if opt.min_ssim else ""}{f",planar={opt.planar_convs}" if opt.planar_convs else ""}{opt.config_tag}'
    elif (opt.mode == 'animation_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_noise_padding' % (opt.input_name[:-4], opt.scale_factor_init)
    elif (opt.mode == 'paint_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_paint/start_scale=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.paint_start_scale)
    elif opt.mode == 'random_samples':
        dir2save = f'{opt.out}/RandomSamples/{opt.input_name[:-4]}/scale_factor={opt.scale_factor_init:.3f},num_layers={opt.num_layer},sim_alpha={opt.sim_alpha:.3f},sim_boundary={opt.sim_boundary},sim_boundary_type={opt.sim_boundary_type},use_attn_g={opt.use_attention_g},use_attn_end_g={opt.use_attention_end_g},use_attn_d={opt.use_attention_d},use_attn_end_d={opt.use_attention_end_d},nfc={opt.nfc_init},min_size={opt.min_size},few_gan={opt.few_gan},sim_cond_d={opt.sim_cond_d},num_layer_d={opt.num_layer_d}{"_groupnorm" if opt.groupnorm else ""}{",prelu" if opt.prelu else ""}{",rel" if opt.relativistic else ""}{",criticGenerator" if opt.generate_with_critic else ""}{"_detach" if opt.detach_critic else ""}{",train_last_longer" if opt.train_last_layer_longer else ""}{",split" if opt.split_image else ""}{",min_ssim" if opt.min_ssim else ""}{f",planar={opt.planar_convs}" if opt.planar_convs else ""}{opt.config_tag}/gen_start_scale={opt.gen_start_scale}'
    elif opt.mode == 'random_samples_arbitrary_sizes':
        dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (opt.out,opt.input_name[:-4], opt.scale_v, opt.scale_h)
    elif opt.mode == 'animation':
        dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
    elif opt.mode == 'SR':
        dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
    elif opt.mode == 'harmonization':
        dir2save = '%s/Harmonization/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'editing':
        dir2save = '%s/Editing/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'paint2image':
        dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
        if opt.quantization_flag:
            dir2save = '%s_quantized' % dir2save
    return dir2save

def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else f"cuda:{opt.device}")
    if not opt.not_cuda:
        torch.cuda.set_device(opt.device)
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor)
    if opt.mode == 'SR':
        opt.alpha = 100

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

def calc_init_scale(opt):
    in_scale = math.pow(1/2,1/3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale,iter_num

def quant(prev,device):
    arr = prev.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if () else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor.to(device))
    x = x.view(prev.shape)
    return x,centers

def quant2centers(paint, centers):
    arr = paint.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, init=centers, n_init=1).fit(arr)
    labels = kmeans.labels_
    #centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if torch.cuda.is_available() else x.type(torch.FloatTensor)
    #x = x.type(torch.cuda.FloatTensor)
    x = x.view(paint.shape)
    return x

    return paint


def dilate_mask(mask,opt):
    if opt.mode == "harmonization":
        element = morphology.disk(radius=7)
    if opt.mode == "editing":
        element = morphology.disk(radius=20)
    mask = torch2uint8(mask)
    mask = mask[:,:,0]
    mask = morphology.binary_dilation(mask,selem=element)
    mask = filters.gaussian(mask, sigma=5)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2torch(mask,opt)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    plt.imsave('%s/%s_mask_dilated.png' % (opt.ref_dir, opt.ref_name[:-4]), convert_image_np(mask), vmin=0,vmax=1)
    mask = (mask-mask.min())/(mask.max()-mask.min())
    return mask


