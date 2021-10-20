import os
import cv2
import torch
import shutil
import random
import numpy as np

from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

def pathExistenceTest(path, delete=False, contain=False):
    '''
    for log_dir: if exist, then delete it's files and folders under it, if not, make it;
    for result_dir: if not exist, make it.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    elif delete:
        deleteUnder(path, contain=contain)
        
def deleteUnder(path, contain=False):
    '''
    delete all files and folders under path
    :param path: Folder to be deleted
    :param contain: delete root or not
    '''
    if contain:
        shutil.rmtree(path)
    else:
        del_list = os.listdir(path)
        for f in del_list:
            file_path = os.path.join(path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                
def printParaNum(model):
    
    '''
    function: print the number of total parameters and trainable parameters
    '''
    
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters: %d' % total_params)
    print('trainable parameters: %d' % total_trainable_params)


class LambdaLR:
    def __init__(self, n_epochs, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

def getMeanMetrics(tensor_image1, tensor_image2, psnr_only=False, reduction=True):
    
    '''
    function: given a batch tensor image pair, get the mean or sum psnr and ssim value.
    input:  range:[-1,1]     type:tensor.FloatTensor  format:[b,c,h,w]  RGB
    output: two python value, e.g., psnr_value, ssim_value
    '''
    
    if len(tensor_image1.shape) != 4 or len(tensor_image2.shape) != 4:
        raise Excpetion('a batch tensor image pair should be given!')
        
    numpy_imgs = denormalize(tensor_image1)
    numpy_gts = denormalize(tensor_image2)
    psnr_value, ssim_value = 0., 0.
    batch_size = numpy_imgs.shape[0]
    for i in range(batch_size):
        if not psnr_only:
            ssim_value += structural_similarity(numpy_imgs[i],numpy_gts[i], multichannel=True, gaussian_weights=True, use_sample_covariance=False)
        psnr_value += peak_signal_noise_ratio(numpy_imgs[i],numpy_gts[i])
        
    if reduction:
        psnr_value = psnr_value/batch_size
        ssim_value = ssim_value/batch_size
    
    if not psnr_only:  
        return psnr_value, ssim_value
    else:
        return psnr_value
    
def CHRMSE_CRR(tensor_image1, tensor_image2, reduction=True):
    
    '''
    function: given a batch tensor image pair, get the mean or sum chrmse and crr value.
    input:  range:[-1,1]     type:tensor.FloatTensor  format:[b,c,h,w]  RGB
    output: two python value, e.g., chrmse_value, crr_value
    '''
    
    if len(tensor_image1.shape) != 4 or len(tensor_image2.shape) != 4:
        raise Excpetion('a batch tensor image pair should be given!')
        
    numpy_imgs = denormalize(tensor_image1)
    numpy_gts = denormalize(tensor_image2)  
    chrmse_value, crr_value = 0., 0.
    batch_size = numpy_imgs.shape[0]
    
    for i in range(batch_size):
        error_m1_rss = 0
        sqrt_rss = 0
        error_m1_tss = 0
        for j in range(3):
            if j == 0: # 'H' of 'HSV'
                hist1 = cv2.calcHist([cv2.cvtColor(numpy_imgs[i],cv2.COLOR_BGR2HSV)], [j], None, [180], [0, 180])
                hist2 = cv2.calcHist([cv2.cvtColor(numpy_gts[i],cv2.COLOR_BGR2HSV)], [j], None, [180], [0, 180])
            else: #'S' or 'V' of 'HSV'
                hist1 = cv2.calcHist([cv2.cvtColor(numpy_imgs[i],cv2.COLOR_BGR2HSV)], [j], None, [256], [0, 256])
                hist2 = cv2.calcHist([cv2.cvtColor(numpy_gts[i],cv2.COLOR_BGR2HSV)], [j], None, [256], [0, 256])

            hist3 = np.ones(hist2.shape)*np.mean(hist2)
            error_m1_rss += mean_squared_error(hist1, hist2)
            sqrt_rss += np.sqrt(error_m1_rss)
            error_m1_tss += mean_squared_error(hist2, hist3)

        chrmse_value += sqrt_rss/3
        crr_value += max(0, 1-error_m1_rss/error_m1_tss)
        
    if reduction:
        chrmse_value = chrmse_value/batch_size
        crr_value = crr_value/batch_size
    
    return chrmse_value, crr_value

def sample_image(nrow, image_dir, epoch, i, real_A, real_B, fake_A, fake_B, residual_A, residual_B):
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=nrow, normalize=True)
    real_B = make_grid(real_B, nrow=nrow, normalize=True)
    fake_A = make_grid(fake_A, nrow=nrow, normalize=True)
    fake_B = make_grid(fake_B, nrow=nrow, normalize=True)
    residual_A = make_grid(residual_A, nrow=nrow, normalize=True)
    residual_B = make_grid(residual_B, nrow=nrow, normalize=True)
    image_grid = torch.cat((real_A, fake_B, residual_B, real_B, fake_A, residual_A), 1)
                    
    save_image(image_grid, image_dir + "/epoch_{:0>3}_iter_{:0>4}.png".format(epoch+1,i+1), normalize=False)
                                         
def denormalize(tensor_image):
    
    '''
    function: transform a tensor image to a numpy image
    input:  range:[-1,1]     type:tensor.FloatTensor  format:[b,c,h,w]  RGB
    output: range:[0,255]    type:numpy.uint8         format:[b,h,w,c]  RGB
    '''
    
    tensor_image = (tensor_image+1)/2*255
    if tensor_image.device != 'cpu':
        tensor_image = tensor_image.cpu()
    numpy_image = np.transpose(tensor_image.numpy(), (0, 2, 3, 1)) 
    numpy_image = np.uint8(numpy_image)
    return numpy_image