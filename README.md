# CRNet
## Model
train:

`python train.py --data_source /path/to/dataset --trial trial --lambda_perc 0.1 --PGBFP`

test:

`python test.py --data_source /path/to/dataset --trial trial --model_name your_model_name --PGBFP`

## Color-Sensitive Error (CSE) Metric
Example:

```

  import cv2
  from CSE import *
     
  img1 = cv2.imread('img path1')
  img2 = cv2.imread('img path2')
  cse = CSE(img1, img2)
  
```

**Note: the format of image should be BGR, not RGB!**
