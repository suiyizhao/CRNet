# CRNet
## Model
train:

`python train.py --data_source /path/to/dataset --trial trial --resizeX 128 --resizeY 224 --cropX 128 --cropY 128 --lambda_perc 0.1 --PGBFP`

test:

`python test.py --data_source /path/to/dataset --trial trial --resizeX 128 --resizeY 224 --model_name your_model_name --PGBFP`

## CSE Metric
Example:

```

  import cv2
  from CSE import *
     
  img1 = cv2.imread('img path1')
  img2 = cv2.imread('img path2')
  cse = CSE(img1, img2)
  
```

**Note: the format of image should be BGR, not RGB!**
