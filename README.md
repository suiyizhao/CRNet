# CRNet
## Color-Retention Network
### Dependencies
- python 3.6.8
- torch 1.3.0
- tqdm 4.36.1
- numpy 1.17.3
- pillow 8.4.0
- torchvision 0.4.1
- scikit-image 0.17.2
- tensorboardX 1.9
- opencv-python 4.1.1.26
### Datasets
- [GoPro](https://seungjunnah.github.io/Datasets/gopro) (train and test using blur_gamma)
- [CelebA](https://link.zhihu.com/?target=http%3A//mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
### Usage
#### Prepare dataset:
Please ensure that the data organization matches the [code format](https://github.com/suiyizhao/CRNet/blob/master/src/datasets.py#:~:text=sorted(glob.glob(-,os.path.join(opt.data_source%2C%20%22%25s/blurry%22%20%25%20mode)%20%2B%20%22/*/*.*%22,-))).
#### Train:
`python train.py --data_source /path/to/dataset --trial trial --lambda_perc 0.1 --PGBFP`
#### Test:
`python test.py --data_source /path/to/dataset --trial trial --model_name your_model_name --PGBFP`
### PSNR&SSIM

```
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

img1 = cv2.imread('img path1')
img2 = cv2.imread('img path2')
psnr = peak_signal_noise_ratio(img1, img2)
ssim = structural_similarity(img1, img2, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
```

## New Metric: Color-Sensitive Error (CSE)
Usage (CSE.py):

```

  import cv2
  from CSE import *
     
  img1 = cv2.imread('img path1')
  img2 = cv2.imread('img path2')
  cse = CSE(img1, img2)
  
```

**Note: the format of image should be BGR, not RGB!**
