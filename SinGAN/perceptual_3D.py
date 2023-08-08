import torch
from torch import nn
from torch.nn import functional as F
from SinGAN import resnet_medicalnet

def normalize_intensity(volume):
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    out_random = torch.normal(0, 1, size = volume.shape).cuda()
    out[volume == 0] = out_random[volume == 0]
    return out

class MedicalNetLoss(nn.Module):
    """Computes the perceptual loss between two batches of images.
    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes.
    """

    models = {'resnet_10': resnet_medicalnet.resnet10, 
              'resnet_18': resnet_medicalnet.resnet18,
              'resnet_34': resnet_medicalnet.resnet34,
              'resnet_50': resnet_medicalnet.resnet50}

    def __init__(self, model='resnet_50', layer=2, reduction='mean', normalize=True):
        super().__init__()
        self.reduction = reduction
        print("loading model",model)
        self.model = self.models[model](num_seg_classes=1).cuda()
        self.model.eval()
        self.model.requires_grad_(False)
        self.layer = layer
        self.normalize = normalize

        # Load the pretrained model
        net_dict = self.model.state_dict()
        checkpoint = torch.load(f'../data/MedicalNet_pretrain/{model}_23dataset.pth')
        remove_prefix = "module."
        pretrain_dict = {k.replace(remove_prefix, ""): v for k, v in checkpoint['state_dict'].items() \
                         if k.replace(remove_prefix, "") in net_dict.keys()}    
        assert len(pretrain_dict.keys()) > 0    
        net_dict.update(pretrain_dict)
        self.model.load_state_dict(net_dict)

    def forward(self, input, target):
        if self.normalize:
            input = normalize_intensity(input)
            target = normalize_intensity(target)
        input_feats = self.model(input, layer_stop = self.layer)
        target_feats = self.model(target, layer_stop = self.layer)
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)