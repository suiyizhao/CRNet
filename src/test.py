import os
import time
import tqdm
import torch
import pickle
import itertools
import skimage.io as io
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from loss import *
from utils import *
from models import *
from options import *
from datasets import *

print('---------------------------------------- step 1/4 : parameters preparing... ----------------------------------------')
opt = TestOptions().parse()

output_dir = os.path.join(opt.output_dir, opt.trial)

# Create sample and checkpoint directories
pathExistenceTest(output_dir, delete=True)
pathExistenceTest(output_dir + '/single/blurry', delete=True) 
pathExistenceTest(output_dir + '/single/CRNet', delete=True) 
pathExistenceTest(output_dir + '/single/sharp', delete=True) 
pathExistenceTest(opt.pretrained_dir, delete=False)

print('---------------------------------------- step 2/4 : data loading... ------------------------------------------------')
dataset = ImageDataset(opt, unaligned=False, mode="test")

# create dataloader
val_loader = DataLoader(dataset, batch_size=opt.val_size, shuffle=False, num_workers=opt.n_cpus)

# Convert val_loader to a list to save time
# val_loader = list(val_loader)
print('image: ', len(val_loader))

print('---------------------------------------- step 3/4 : model defining... ----------------------------------------------')
# Initialize generator and discriminator
G_AB = Generator(opt.PGBFP, opt.n_blocks).cuda()

print('G:', end = '')
printParaNum(G_AB)

G_AB.load_state_dict(torch.load(opt.pretrained_dir + '/' + opt.model_name +'_G_AB.pth'))

print('---------------------------------------- step 4/4 : testing... ----------------------------------------------------')
def main():
    G_AB.eval()
    
    psnr_value = 0.
    ssim_value = 0.
    for i, batch in enumerate(val_loader):
        real_A = batch["A"].cuda()
        real_B = batch["B"].cuda()
        
        # foward
        fake_B, residual_B = G_AB(real_A)
        psnr, ssim = getMeanMetrics(fake_B.detach(), real_B, psnr_only=False, reduction=False)
        print('iter: {} psnr: {:.4f} ssim: {:.4f}'.format(i, psnr, ssim))
        psnr_value += psnr
        ssim_value += ssim
        save_image(make_grid([real_A[0],fake_B[0],residual_B[0],real_B[0]], nrow=4, normalize=True, scale_each=True), output_dir + '/iter_' + str(i) + '.png')
        io.imsave(output_dir + '/single/blurry/iter_' + str(i) + '.png', denormalize(real_A).squeeze())
        io.imsave(output_dir + '/single/CRNet/iter_' + str(i) + '.png', denormalize(fake_B.detach()).squeeze())
        io.imsave(output_dir + '/single/sharp/iter_' + str(i) + '.png', denormalize(real_B).squeeze())
    
    print(' ')
    print('Total: psnr: {:.4f} ssim: {:.4f}'.format(psnr_value / len(val_loader), ssim_value / len(val_loader)))
    print(' ')

if __name__ == '__main__':
    main()