# ERA V1 Session 9 Assignment

# Data Exploration
CIFAR-10 contains 1000 images per class for test, and 5000 images per class for train.<br>
The classes on CIFAR-10 are Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.<br>

### Some sample images from train set -- 
![image](https://github.com/GunaKoppula/ERA-V1---Session-9/assets/61241928/47e27161-a019-4aa6-9136-948f4f5cdc09)


# Model Architecture
Our architecture is C1C2C3C40 without using any pooling operations.<br>
Instead of pooling we used a combination of dilated convolutions and strided covolutions. <br>
In C3 we have also used depthwise seperable convolutions instead of normal covolution layers.<br>
The total number of parameters for our model is 181,984. <br>
We have not used any Dense Layer, Instead we targeted GAP to get the output classes dim.

```python
        x = self.convblock_1(x)
        x = self.convblock_2(x)
        x = self.dilated_conv_1(x)

        x = self.convblock_3(x)
        x = self.convblock_4(x)
        x = self.dilated_conv_2(x)
        
        x = self.sep_conv_1(x)
        x = self.sep_conv_2(x)
        x = self.strided_conv_1(x)
        
        x = self.convblock_5(x)
        x = self.convblock_6(x)

        x = self.gap(x)
        x = x.view(-1, 10)
```

## Model Summary

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
           Dropout-3           [-1, 32, 32, 32]               0
       BatchNorm2d-4           [-1, 32, 32, 32]              64
            Conv2d-5           [-1, 64, 32, 32]          18,432
              ReLU-6           [-1, 64, 32, 32]               0
           Dropout-7           [-1, 64, 32, 32]               0
       BatchNorm2d-8           [-1, 64, 32, 32]             128
            Conv2d-9           [-1, 64, 32, 32]          36,864
             ReLU-10           [-1, 64, 32, 32]               0
          Dropout-11           [-1, 64, 32, 32]               0
      BatchNorm2d-12           [-1, 64, 32, 32]             128
           Conv2d-13           [-1, 32, 17, 17]           2,080
             ReLU-14           [-1, 32, 17, 17]               0
      BatchNorm2d-15           [-1, 32, 17, 17]              64
           Conv2d-16           [-1, 64, 17, 17]          18,432
             ReLU-17           [-1, 64, 17, 17]               0
          Dropout-18           [-1, 64, 17, 17]               0
      BatchNorm2d-19           [-1, 64, 17, 17]             128
           Conv2d-20           [-1, 64, 17, 17]          36,864
             ReLU-21           [-1, 64, 17, 17]               0
          Dropout-22           [-1, 64, 17, 17]               0
      BatchNorm2d-23           [-1, 64, 17, 17]             128
           Conv2d-24             [-1, 32, 9, 9]           2,080
             ReLU-25             [-1, 32, 9, 9]               0
      BatchNorm2d-26             [-1, 32, 9, 9]              64
           Conv2d-27             [-1, 32, 9, 9]              32
           Conv2d-28             [-1, 64, 9, 9]           2,048
             ReLU-29             [-1, 64, 9, 9]               0
      BatchNorm2d-30             [-1, 64, 9, 9]             128
  SeparableConv2d-31             [-1, 64, 9, 9]               0
           Conv2d-32             [-1, 64, 9, 9]          36,864
             ReLU-33             [-1, 64, 9, 9]               0
          Dropout-34             [-1, 64, 9, 9]               0
      BatchNorm2d-35             [-1, 64, 9, 9]             128
           Conv2d-36             [-1, 32, 6, 6]           2,080
             ReLU-37             [-1, 32, 6, 6]               0
      BatchNorm2d-38             [-1, 32, 6, 6]              64
           Conv2d-39             [-1, 64, 6, 6]          18,432
             ReLU-40             [-1, 64, 6, 6]               0
          Dropout-41             [-1, 64, 6, 6]               0
      BatchNorm2d-42             [-1, 64, 6, 6]             128
           Conv2d-43             [-1, 10, 6, 6]           5,760
        AvgPool2d-44             [-1, 10, 1, 1]               0
================================================================
Total params: 181,984
Trainable params: 181,984
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.84
Params size (MB): 0.69
Estimated Total Size (MB): 7.54
----------------------------------------------------------------
```

# Augmentation

```python
        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
               shift_limit=0.0625, scale_limit=0.1, 
                rotate_limit=45, interpolation=1, 
                border_mode=4, p=0.5
            ),
            A.CoarseDropout(
                max_holes=2, max_height=8, 
                max_width=8, p=0.3
            ),
            A.RandomBrightnessContrast(p=0.2),
            A.ToGray(p=0.1),
            A.Normalize(
                mean=self.mean, 
                std=self.std,
                always_apply=True
            ),
            ToTensorV2()
        ])
```


# Goals 

- [X] Model is trained on GPU
- [X] change the architecture to C1C2C3C40  (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
- [X] total RF must be more than 44.
- [X] one of the layers must use Depthwise Separable Convolution. _(Bonus points for two layers)_
- [X] one of the layers must use Dilated Convolution
- [X] use GAP (compulsory):- add FC after GAP to target #of classes (optional) _(if optional achieved Bonus points)_
- [X] use albumentation library and apply:
  - [X] horizontal flip
  - [X] shiftScaleRotate
  - [x] coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
- [X] achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.  _(Bonus for 87% acc, and <100k params)_
- [X] upload to Github
