import os
import torch
import torch.nn as nn
import torchvision.models as models

class CycleLoss():
    def __init__(self, criterion):
        self.criterion = criterion
    
    def __call__(self, img1, img2):
        loss = self.criterion(img1, img2)
        return loss
    
class IdentityLoss():
    def __init__(self, criterion):
        self.criterion = criterion
    
    def __call__(self, img1, img2):
        loss = self.criterion(img1, img2)
        return loss

class PerceptualLoss():
    def __init__(self, criterion, layer, path):
        vgg_path = path + '/vgg19.pth'
        if not os.path.exists(vgg_path):
            self.model = models.vgg19(pretrained=True)
        else:
            self.model = models.vgg19(pretrained=False)
            self.model.load_state_dict(torch.load(vgg_path))
            
        self.model = self.model.cuda()
        self.model = self.model.features[:layer]
        self.model.eval()
        
        self.criterion = criterion
    
    def __call__(self, img1, img2):
        feature1 = self.model(img1)
        feature2 = self.model(img2)
        loss = self.criterion(feature1, feature2)
        return loss

class GanLoss():
    def __init__(self, gan_type):
        self.gan_type = gan_type
        if self.gan_type == 'vanilla':
            self.criterion = nn.BCELoss()
        elif self.gan_type == 'lsgan':
            self.criterion = nn.MSELoss()
        
    def __call__(self, discriminator, fake):
        fake_valid = discriminator(fake)
        loss = 0.
        for i in range(len(fake_valid)):
            all1 = torch.ones_like(fake_valid[i]).cuda()
            if self.gan_type == 'vanilla':
                real_loss = self.criterion(torch.sigmoid(fake_valid[i]), all1)
            elif self.gan_type == 'lsgan':
                real_loss = self.criterion(fake_valid[i], all1)
            elif self.gan_type == 'wgan-gp':
                real_loss = -torch.mean(fake_valid[i])
            else:
                raise Exception('no such type of gan')
            loss += real_loss
        return loss

class DLoss():
    def __init__(self, gan_type):
        self.gan_type = gan_type
        if self.gan_type == 'vanilla':
            self.criterion = nn.BCELoss()
        elif self.gan_type == 'lsgan':
            self.criterion = nn.MSELoss()
    
    def __call__(self, discriminator, fake, real):
        fake_valid = discriminator(fake)
        real_valid = discriminator(real)
        if self.gan_type == 'wgan-gp':
            gp = compute_gradient_penalty(discriminator, fake, real)
        loss = 0.
        for i in range(len(fake_valid)):
            all0 = torch.zeros_like(fake_valid[i]).cuda()
            all1 = torch.ones_like(real_valid[i]).cuda()
            if self.gan_type == 'vanilla':
                fake_loss = self.criterion(torch.sigmoid(fake_valid[i]), all0)
                real_loss = self.criterion(torch.sigmoid(real_valid[i]), all1)
                total_loss = (fake_loss + real_loss) / 2
            elif self.gan_type == 'lsgan':
                fake_loss = self.criterion(fake_valid[i], all0)
                real_loss = self.criterion(real_valid[i], all1)
                total_loss = (fake_loss + real_loss) / 2
            elif self.gan_type == 'wgan-gp':
                total_loss = -torch.mean(real_valid[i]) + torch.mean(fake_valid[i]) + 10*gp[i]
            else:
                raise Exception('no such type of gan')
            loss += total_loss
        return loss

def compute_gradient_penalty(D, fake_samples, real_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).cuda()
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    avgpool = nn.AvgPool2d(2)
    
    gradient_penalty = []
    for i in range(D.n_scales):
        d_interpolates = D.forward_i(i, interpolates)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty.append(((gradients.norm(2, dim=1) - 1) ** 2).mean())
        if i < D.n_scales-1:
            if D.scale_type == 'resize':
                interpolates = avgpool(interpolates)
            elif D.scale_type == 'crop':
                interpolates = interpolates[:,:,int(interpolates.shape[-2]/4):int(3*interpolates.shape[-2]/4),int(interpolates.shape[-1]/4):int(3*interpolates.shape[-1]/4)]
            else:
                raise Exception('no such scale_type')
    return gradient_penalty

