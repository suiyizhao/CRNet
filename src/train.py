import os
import time
import tqdm
import torch
import pickle
import itertools
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from loss import *
from utils import *
from models import *
from options import *
from datasets import *

print('---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------')
opt = TrainOptions().parse()

model_dir = os.path.join(opt.results_dir, opt.trial, 'models')
image_dir = os.path.join(opt.results_dir, opt.trial, 'images')
log_dir = os.path.join(opt.log_dir, opt.trial)

# Create sample and checkpoint directories
if not opt.resume:
    pathExistenceTest(model_dir, delete=True)
    pathExistenceTest(image_dir, delete=True)
    pathExistenceTest(log_dir, delete=True)
    
pathExistenceTest(opt.pretrained_dir, delete=False)

writer = SummaryWriter(log_dir=log_dir)

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

latest_psnr = [0.]
cur_patience = [0]

print('---------------------------------------- step 2/5 : data loading... ------------------------------------------------')
# create dataset
train_dataset = ImageDataset(opt, unaligned=True)
val_dataset = ImageDataset(opt, unaligned=False, mode="test")

# create dataloader
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpus)
val_loader = DataLoader(val_dataset, batch_size=opt.val_size, shuffle=False, num_workers=opt.n_cpus)

# Convert val_loader to a list to save time
val_loader = list(val_loader)

print('---------------------------------------- step 3/5 : model defining... ----------------------------------------------')
# Initialize generator and discriminator
G_AB = Generator(opt.PGBFP, opt.n_blocks).cuda()
G_BA = Generator(opt.PGBFP, opt.n_blocks).cuda()
D_A = Discriminator(opt.n_scales, opt.scale_type).cuda()
D_B = Discriminator(opt.n_scales, opt.scale_type).cuda()

print('G:', end = '')
printParaNum(G_AB)
print('D:', end = '')
printParaNum(D_A)

if opt.resume:
    with open(os.path.join(model_dir, 'checkpoint_dict.pth'), 'rb') as f:
        checkpoint_dict = pickle.load(f)
    opt.start_epoch = checkpoint_dict['epoch'] + 1
    latest_psnr = checkpoint_dict['latest_psnr']
    cur_patience = checkpoint_dict['cur_patience']
    
    G_AB.load_state_dict(torch.load(model_dir + '/latest_params_G_AB.pth'))
    G_BA.load_state_dict(torch.load(model_dir + '/latest_params_G_BA.pth'))
    D_A.load_state_dict(torch.load(model_dir + '/latest_params_D_A.pth'))
    D_B.load_state_dict(torch.load(model_dir + '/latest_params_D_B.pth'))
    
if opt.pretrain:
    G_AB.load_state_dict(torch.load(opt.pretrained_dir + '/' + opt.model_name +'_G_AB.pth'))
    G_BA.load_state_dict(torch.load(opt.pretrained_dir + '/' + opt.model_name +'_G_BA.pth'))
    D_A.load_state_dict(torch.load(opt.pretrained_dir + '/' + opt.model_name +'_D_A.pth'))
    D_B.load_state_dict(torch.load(opt.pretrained_dir + '/' + opt.model_name +'_D_B.pth'))

print('---------------------------------------- step 4/5 : requisites defining... -----------------------------------------')
# Losses
criterion_cycle = CycleLoss(criterion=nn.L1Loss())
criterion_perc = PerceptualLoss(criterion=nn.MSELoss(), layer=28, path=opt.pretrained_dir)
criterion_identity = IdentityLoss(criterion=nn.L1Loss())
criterion_gan = GanLoss(gan_type=opt.gan_type)
criterion_D = DLoss(gan_type=opt.gan_type)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.decay_epoch).step)

print('---------------------------------------- step 5/5 : training... ----------------------------------------------------')
def main():
    for epoch in range(opt.start_epoch, opt.n_epochs):
        print("--------------------------------------")
        print("adaptive factor:  pos: {:.4f} neg: {:.4f}".format(G_AB.get_factor().item(), G_BA.get_factor().item()))
        print("--------------------------------------")
        train(epoch)
        
        if (epoch+1)%opt.val_gap == 0:
            val(epoch)
        
        if cur_patience[0] == opt.patience:
            print('Early stopping at epoch ' + str(epoch+1))
            break
    
def train(epoch):
    G_AB.train()
    G_BA.train()
    
    CRITIC_ITERS = 5 if opt.gan_type == 'wgan-gp' else 1
    time_start = time.time()
    num, loss_D_value, loss_G_value, loss_gan_value, loss_cycle_value, loss_perc_value, loss_identity_value = 0., 0., 0., 0., 0., 0., 0.
    for i, batch in enumerate(train_loader):
        
        # Set model input
        real_A = batch["A"].cuda()
        real_B = batch["B"].cuda()
        
        # -----------------------
        #  Train Discriminator
        # -----------------------
        
        # foward
        fake_B, _ = G_AB(real_A)
        fake_A, _ = G_BA(real_B)
        
        # -----------------------
        #  Train Discriminator A
        optimizer_D_A.zero_grad()

        # compute loss
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_D_A = criterion_D(D_A, fake_A_.detach(), real_A)

        # backward & update
        loss_D_A.backward()
        optimizer_D_A.step()
        
        # -----------------------
        #  Train Discriminator B
        optimizer_D_B.zero_grad()

        # compute loss
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_D_B = criterion_D(D_B, fake_B_.detach(), real_B)

        # backward & update
        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2
        
        if (i+1) % CRITIC_ITERS == 0:
            # ------------------
            #  Train Generator
            # ------------------

            # avoid updating D parameters
            for p in D_A.parameters():
                p.requires_grad = False
            for p in D_B.parameters():
                p.requires_grad = False

            optimizer_G.zero_grad()

            # foward
            iden_B, _ = G_AB(real_B)
            iden_A, _ = G_BA(real_A)
            recov_B, _ = G_AB(fake_A)
            recov_A, _ = G_BA(fake_B)

            # compute loss
            loss_cycle = (criterion_cycle(recov_B, real_B) + criterion_cycle(recov_A, real_A)) / 2
            loss_perc = (criterion_perc(fake_B, real_A) + criterion_perc(fake_A, real_B)) / 2
            loss_identity = (criterion_identity(iden_B, real_B) + criterion_identity(iden_A, real_A)) / 2
            loss_gan = (criterion_gan(D_B, fake_B) + criterion_gan(D_A, fake_A)) / 2

            loss_G = loss_gan + opt.lambda_cycle * loss_cycle + opt.lambda_perc * loss_perc + opt.lambda_identity * loss_identity
            
            # backward & update
            loss_G.backward()
            optimizer_G.step()

            # unfreeze update D parameters
            for p in D_A.parameters():
                p.requires_grad = True
            for p in D_B.parameters():
                p.requires_grad = True
        
            # --------------
            #  Log Progress
            # --------------

            num += real_B.shape[0]
            loss_D_value += loss_D.item()
            loss_G_value += loss_G.item()
            loss_gan_value += loss_gan.item()
            loss_identity_value += loss_identity.item()
            loss_perc_value += loss_perc.item()
            loss_cycle_value += loss_cycle.item()
        
        if (i+1) % opt.show_gap == 0:
            time_end = time.time()
            times = time_end-time_start
            time_start = time_end
            
            print('Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>4}/{:0>4}] Loss_D: {:.4f} Loss_G: {:.4f} Loss_gan: {:.4f} Loss_cycle: {:.4f} Loss_perc: {:.4f} Loss_identity: {:.4f} Time:{:.4f}(s)'.format(
                       epoch + 1, opt.n_epochs, i + 1, len(train_loader), loss_D_value / num, loss_G_value / num, loss_gan_value / num, loss_cycle_value / num, loss_perc_value / num, loss_identity_value / num, times))
            
            writer.add_scalar('d_loss', loss_D_value / num, i + len(train_loader)*epoch)
            writer.add_scalar('g_loss', loss_G_value / num, i + len(train_loader)*epoch) 
            writer.add_scalar('gan_loss', loss_gan_value / num, i + len(train_loader)*epoch)
            writer.add_scalar('cycle_loss', loss_cycle_value / num, i + len(train_loader)*epoch)
            writer.add_scalar('perc_loss', loss_perc_value / num, i + len(train_loader)*epoch)
            writer.add_scalar('identity_loss', loss_identity_value / num, i + len(train_loader)*epoch)
            
            num, loss_D_value, loss_G_value, loss_gan_value, loss_cycle_value, loss_perc_value, loss_identity_value = 0., 0., 0., 0., 0., 0., 0.
            
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    
    save('checkpoint', epoch=epoch)
    
def val(epoch):
    G_AB.eval()
    G_BA.eval()
    
    show_num = 0
    show_random = random.randrange(1, len(val_loader))
    
    num, psnr_value = 0., 0.
    time_start = time.time()
    for i, batch in tqdm.tqdm(enumerate(val_loader)):
        real_A = batch["A"].cuda()
        real_B = batch["B"].cuda()
        
        # foward
#         print(real_A.shape)
        fake_B, residual_B = G_AB(real_A)
        fake_A, residual_A = G_BA(real_B)
        
        num += real_B.shape[0]
        psnr_value += getMeanMetrics(fake_B.detach(), real_B, psnr_only=True, reduction=False)
        
        if (i+1)%show_random == 0 and show_num != 3:      
            
            sample_image(opt.val_size, image_dir, epoch, i, real_A, real_B, fake_A, fake_B, residual_A, residual_B)
            
            show_num += 1
            show_random = random.randrange(i+1, len(val_loader))
        
        # save an image, This number 310 can be any integer in the iteration, 207, 348
        if (i+1) % 348 == 0:
            if (epoch+1) == opt.val_gap:
                save_image(make_grid([real_A[0],real_A[0],torch.zeros(real_A[0].size()).cuda()], nrow=3, normalize=True, scale_each=True), image_dir + '/AnImage_epoch_{:0>3}.png'.format(0))
            save_image(make_grid([real_A[0],fake_B[0],residual_B[0]], nrow=3, normalize=True, scale_each=True), image_dir + '/AnImage_epoch_{:0>3}.png'.format(epoch+1))
            
    writer.add_scalar('psnr', psnr_value / num, epoch)
    writer.add_scalar('lr', lr_scheduler_G.get_lr()[0], epoch)
    
    print(' ')
    print('Validating: Psnr: {:.4f} Time:{:.4f}(s)'.format(psnr_value / num, time.time()-time_start))
    print(' ')
    
    if (epoch+1) % opt.save_gap == 0:
        
        torch.save(G_AB.state_dict(), model_dir + '/G_AB_epoch_{:0>3}.pth'.format(epoch+1))
        torch.save(G_BA.state_dict(), model_dir + '/G_BA_epoch_{:0>3}.pth'.format(epoch+1))
        torch.save(D_A.state_dict(), model_dir + '/D_A_epoch_{:0>3}.pth'.format(epoch+1))
        torch.save(D_B.state_dict(), model_dir + '/D_B_epoch_{:0>3}.pth'.format(epoch+1))
        
    save('optimal_model', val_psnr=psnr_value / num)
    
def save(savetype, val_psnr=None, epoch=None):
    # update model
    if savetype=='optimal_model':
        if val_psnr > latest_psnr[0]:
            latest_psnr[0] = val_psnr

            torch.save(G_AB.state_dict(), model_dir + '/optimal_params_{:.3f}_G_AB'.format(latest_psnr[0])+'.pth')
            torch.save(G_BA.state_dict(), model_dir + '/optimal_params_{:.3f}_G_BA'.format(latest_psnr[0])+'.pth')
            torch.save(D_A.state_dict(), model_dir + '/optimal_params_{:.3f}_D_A'.format(latest_psnr[0])+'.pth')
            torch.save(D_B.state_dict(), model_dir + '/optimal_params_{:.3f}_D_B'.format(latest_psnr[0])+'.pth')
            
            cur_patience[0] = 0
        else:
            cur_patience[0] += 1
    elif savetype=='checkpoint':
            
            torch.save(G_AB.state_dict(), model_dir + '/latest_params_G_AB.pth')
            torch.save(G_BA.state_dict(), model_dir + '/latest_params_G_BA.pth')
            torch.save(D_A.state_dict(), model_dir + '/latest_params_D_A.pth')
            torch.save(D_B.state_dict(), model_dir + '/latest_params_D_B.pth')
            
            checkpoint_dict = {'latest_psnr':latest_psnr, 'cur_patience':cur_patience, 'epoch':epoch}
            with open(os.path.join(model_dir, 'checkpoint_dict.pth'), 'wb') as f:
                pickle.dump(checkpoint_dict, f, pickle.HIGHEST_PROTOCOL)
    else:
        raise Exception('No such savetype in function save()!')

if __name__ == '__main__':
    main()