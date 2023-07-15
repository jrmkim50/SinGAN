import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--mode', help='task to be done', default='train')
    #workspace:
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--device', type=int, help='which cuda device?', default=0)
    
    #load, input, save configurations:
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nc_z',type=int,help='noise # channels',default=1)
    parser.add_argument('--nc_im',type=int,help='image # channels',default=1)
    parser.add_argument('--out',help='output folder',default='Output')
    # parser.add_argument('--config_tag',type=str,required=True)
        
    #networks hyper parameters:
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--min_nfc', type=int, default=32)
    parser.add_argument('--ker_size',type=int,help='kernel size',default=3)
    parser.add_argument('--num_layer',type=int,help='number of layers',default=5)
    parser.add_argument('--num_layer_d',type=int, help='number of layers in discrim',default=0)
    parser.add_argument('--stride',help='stride',default=1)
    parser.add_argument('--padd_size',type=int,help='net pad size',default=0)#math.floor(opt.ker_size/2)
    parser.add_argument('--discrim_no_spatial',type=int, help='0 to use spatial attention in discriminator',default=1)
        
    #pyramid parameters:
    parser.add_argument('--scale_factor',type=float,help='pyramid scale factor',default=0.75)#pow(0.5,1/6))
    parser.add_argument('--noise_amp',type=float,help='addative noise cont weight',default=0.1)
    parser.add_argument('--min_size',type=int,help='image minimal size at the coarser scale',default=25)
    parser.add_argument('--max_size', type=int,help='image minimal size at the coarser scale', default=250)

    #optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train per scale')
    parser.add_argument('--gamma',type=float,help='scheduler gamma',default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--Gsteps',type=int, help='Generator inner steps',default=3)
    parser.add_argument('--Dsteps',type=int, help='Discriminator inner steps',default=3)
    parser.add_argument('--lambda_grad',type=float, help='gradient penelty weight',default=0.1)
    parser.add_argument('--alpha',type=float, help='reconstruction loss weight',default=10)
    # parser.add_argument('--linear_sim',type=int, help='0 for no linear, 1 for decreasing linear, 0 for increasing linear',default=0)
    parser.add_argument('--sim_alpha',type=float, help='simularity loss weight',default=10)
    parser.add_argument('--sim_boundary',type=int, help='Apply sim loss from pyramid layers [sim_boundary, num_stages) or [0,sim_boundary]',default=3)
    parser.add_argument('--sim_boundary_type',type=str, help='Is the boundary a start or an end?',default='start')
    parser.add_argument('--sim_type',type=str, help='What type of sim loss?',default='vgg')
    parser.add_argument('--use_attention_g',type=int, help='Use attention?',default=1)
    parser.add_argument('--use_attention_end_g',type=int, help='Use attention?',default=1)
    parser.add_argument('--use_attention_d',type=int, help='Use attention?',default=1)
    parser.add_argument('--use_attention_end_d',type=int, help='Use attention?',default=1)

    parser.add_argument('--few_gan',type=int, help='Number of reference images',default=0) # Name credit to Cole Kissane!
    parser.add_argument('--config_tag',type=str, help='extra identifying info',default="")

    parser.add_argument('--groupnorm', action='store_true', help='use groupnorm', default=0)
    parser.add_argument('--prelu', action='store_true', help='use prelu', default=0)
    parser.add_argument('--relativistic', action='store_true', help='use relativistic discrim', default=0)
    parser.add_argument('--train_last_layer_longer', action='store_true', help='train last scale for 2 * original num iters', default=0)
    parser.add_argument('--train_first_layers_longer', type=int, help='train first n scales for 2 * original num iters', default=0)

    parser.add_argument('--split_image', action='store_true', help='fold image in half', default=0)
    parser.add_argument('--harmonic_ssim', action='store_true', help='use harmonic mean ssim value', default=0)

    parser.add_argument('--discrim_no_fewgan', action='store_true', help='only train discriminator with original real image', default=0)

    parser.add_argument('--warmup_g', action='store_true', help='warmup for generator', default=0)
    parser.add_argument('--warmup_d', action='store_true', help='warmup for discriminator', default=0)

    parser.add_argument('--spectral_norm_g', action='store_true', help='spectral norm for generator', default=0)
    parser.add_argument('--spectral_norm_d', action='store_true', help='spectral norm for discriminator', default=0)

    parser.add_argument('--packing_level', type=int, help='packing level from PacGAN', default=0)

    # parser.add_argument('--discrim_recon', action='store_true', help='include random recon image in discriminator loss', default=0) => DID NOT WORK
    # parser.add_argument('--min_ssim', action='store_true', help='use the minimum ssim value', default=0) => DID NOT WORK
    
    return parser
