# ERA V1 Session 9 Assignment

# Data Exploration
CIFAR-10 contains 1000 images per class for test, and 5000 images per class for train.<br>
The classes on CIFAR-10 are Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.<br>

### Some sample images from train set -- 
![image](https://github.com/Navyabhat03/ERA-V1-Session-9/assets/60884505/31583334-5835-4bbc-95bb-d904b68a7af8)



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

```python
[EPOCH 0 / 500] -- 
Loss=1.3323101997375488 Batch_id=390 Accuracy=36.97: 100%|██████████| 391/391 [00:09<00:00, 39.32it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 1.3474, Accuracy: 5111/10000 (51.11%)

[EPOCH 1 / 500] -- 
Loss=1.3395029306411743 Batch_id=390 Accuracy=49.59: 100%|██████████| 391/391 [00:15<00:00, 24.56it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 1.1453, Accuracy: 5814/10000 (58.14%)

[EPOCH 2 / 500] -- 
Loss=1.1850202083587646 Batch_id=390 Accuracy=54.71: 100%|██████████| 391/391 [00:10<00:00, 38.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 1.0423, Accuracy: 6245/10000 (62.45%)

[EPOCH 3 / 500] -- 
Loss=1.1960726976394653 Batch_id=390 Accuracy=58.56: 100%|██████████| 391/391 [00:09<00:00, 41.56it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.9479, Accuracy: 6568/10000 (65.68%)

[EPOCH 4 / 500] -- 
Loss=0.8458970785140991 Batch_id=390 Accuracy=60.97: 100%|██████████| 391/391 [00:09<00:00, 40.65it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.8897, Accuracy: 6800/10000 (68.00%)

[EPOCH 5 / 500] -- 
Loss=1.336883544921875 Batch_id=390 Accuracy=63.09: 100%|██████████| 391/391 [00:11<00:00, 33.47it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.8388, Accuracy: 7047/10000 (70.47%)

[EPOCH 6 / 500] -- 
Loss=0.985329806804657 Batch_id=390 Accuracy=64.63: 100%|██████████| 391/391 [00:10<00:00, 36.89it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.8074, Accuracy: 7150/10000 (71.50%)

[EPOCH 7 / 500] -- 
Loss=0.979566216468811 Batch_id=390 Accuracy=66.04: 100%|██████████| 391/391 [00:09<00:00, 39.32it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.7704, Accuracy: 7271/10000 (72.71%)

[EPOCH 8 / 500] -- 
Loss=0.9942011833190918 Batch_id=390 Accuracy=66.96: 100%|██████████| 391/391 [00:11<00:00, 34.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.7276, Accuracy: 7426/10000 (74.26%)

[EPOCH 9 / 500] -- 
Loss=0.7984920740127563 Batch_id=390 Accuracy=67.92: 100%|██████████| 391/391 [00:10<00:00, 38.32it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.7110, Accuracy: 7489/10000 (74.89%)

[EPOCH 10 / 500] -- 
Loss=0.9223629236221313 Batch_id=390 Accuracy=68.93: 100%|██████████| 391/391 [00:11<00:00, 34.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6877, Accuracy: 7594/10000 (75.94%)

[EPOCH 11 / 500] -- 
Loss=0.8129749298095703 Batch_id=390 Accuracy=69.78: 100%|██████████| 391/391 [00:10<00:00, 38.31it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6862, Accuracy: 7595/10000 (75.95%)

[EPOCH 12 / 500] -- 
Loss=0.6988564729690552 Batch_id=390 Accuracy=70.38: 100%|██████████| 391/391 [00:14<00:00, 26.20it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6664, Accuracy: 7659/10000 (76.59%)

[EPOCH 13 / 500] -- 
Loss=1.0190911293029785 Batch_id=390 Accuracy=71.09: 100%|██████████| 391/391 [00:10<00:00, 38.69it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6556, Accuracy: 7679/10000 (76.79%)

[EPOCH 14 / 500] -- 
Loss=0.8204767107963562 Batch_id=390 Accuracy=71.59: 100%|██████████| 391/391 [00:10<00:00, 37.79it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6227, Accuracy: 7865/10000 (78.65%)

[EPOCH 15 / 500] -- 
Loss=0.7832554578781128 Batch_id=390 Accuracy=72.02: 100%|██████████| 391/391 [00:09<00:00, 40.29it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5944, Accuracy: 7951/10000 (79.51%)

[EPOCH 16 / 500] -- 
Loss=0.926749587059021 Batch_id=390 Accuracy=72.72: 100%|██████████| 391/391 [00:13<00:00, 29.17it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6035, Accuracy: 7913/10000 (79.13%)

[EPOCH 17 / 500] -- 
Loss=0.8714362382888794 Batch_id=390 Accuracy=73.07: 100%|██████████| 391/391 [00:10<00:00, 36.45it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5917, Accuracy: 7962/10000 (79.62%)

[EPOCH 18 / 500] -- 
Loss=0.7912548184394836 Batch_id=390 Accuracy=73.28: 100%|██████████| 391/391 [00:10<00:00, 38.78it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5846, Accuracy: 7989/10000 (79.89%)

[EPOCH 19 / 500] -- 
Loss=0.8119962811470032 Batch_id=390 Accuracy=73.47: 100%|██████████| 391/391 [00:11<00:00, 34.26it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5949, Accuracy: 7942/10000 (79.42%)

[EPOCH 20 / 500] -- 
Loss=0.5307092666625977 Batch_id=390 Accuracy=74.14: 100%|██████████| 391/391 [00:10<00:00, 38.21it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5777, Accuracy: 8005/10000 (80.05%)

[EPOCH 21 / 500] -- 
Loss=0.7694862484931946 Batch_id=390 Accuracy=74.38: 100%|██████████| 391/391 [00:10<00:00, 36.00it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5597, Accuracy: 8063/10000 (80.63%)

[EPOCH 22 / 500] -- 
Loss=0.8409790992736816 Batch_id=390 Accuracy=74.61: 100%|██████████| 391/391 [00:10<00:00, 35.94it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5754, Accuracy: 8018/10000 (80.18%)

[EPOCH 23 / 500] -- 
Loss=0.7617672085762024 Batch_id=390 Accuracy=75.16: 100%|██████████| 391/391 [00:10<00:00, 35.68it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5518, Accuracy: 8106/10000 (81.06%)

[EPOCH 24 / 500] -- 
Loss=0.9656778573989868 Batch_id=390 Accuracy=75.01: 100%|██████████| 391/391 [00:12<00:00, 32.25it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5356, Accuracy: 8185/10000 (81.85%)

[EPOCH 25 / 500] -- 
Loss=0.7409988045692444 Batch_id=390 Accuracy=75.35: 100%|██████████| 391/391 [00:11<00:00, 33.26it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5608, Accuracy: 8093/10000 (80.93%)

[EPOCH 26 / 500] -- 
Loss=0.7079620361328125 Batch_id=390 Accuracy=75.66: 100%|██████████| 391/391 [00:13<00:00, 29.78it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5301, Accuracy: 8179/10000 (81.79%)

[EPOCH 27 / 500] -- 
Loss=0.7607666850090027 Batch_id=390 Accuracy=75.71: 100%|██████████| 391/391 [00:10<00:00, 38.79it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5487, Accuracy: 8166/10000 (81.66%)

[EPOCH 28 / 500] -- 
Loss=0.5682892203330994 Batch_id=390 Accuracy=75.97: 100%|██████████| 391/391 [00:10<00:00, 35.67it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5071, Accuracy: 8260/10000 (82.60%)

[EPOCH 29 / 500] -- 
Loss=0.6973447799682617 Batch_id=390 Accuracy=76.14: 100%|██████████| 391/391 [00:09<00:00, 39.84it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5149, Accuracy: 8244/10000 (82.44%)

[EPOCH 30 / 500] -- 
Loss=0.6295843124389648 Batch_id=390 Accuracy=76.35: 100%|██████████| 391/391 [00:10<00:00, 35.71it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5130, Accuracy: 8249/10000 (82.49%)

[EPOCH 31 / 500] -- 
Loss=0.6708377003669739 Batch_id=390 Accuracy=76.54: 100%|██████████| 391/391 [00:11<00:00, 33.31it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5041, Accuracy: 8275/10000 (82.75%)

[EPOCH 32 / 500] -- 
Loss=0.5666126012802124 Batch_id=390 Accuracy=76.90: 100%|██████████| 391/391 [00:09<00:00, 39.80it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5079, Accuracy: 8277/10000 (82.77%)

[EPOCH 33 / 500] -- 
Loss=0.688528835773468 Batch_id=390 Accuracy=76.90: 100%|██████████| 391/391 [00:09<00:00, 39.52it/s]  
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5185, Accuracy: 8228/10000 (82.28%)

[EPOCH 34 / 500] -- 
Loss=0.8061864972114563 Batch_id=390 Accuracy=77.01: 100%|██████████| 391/391 [00:09<00:00, 39.47it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4974, Accuracy: 8291/10000 (82.91%)

[EPOCH 35 / 500] -- 
Loss=0.643330991268158 Batch_id=390 Accuracy=77.47: 100%|██████████| 391/391 [00:09<00:00, 40.89it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4950, Accuracy: 8304/10000 (83.04%)

[EPOCH 36 / 500] -- 
Loss=0.8253556489944458 Batch_id=390 Accuracy=77.29: 100%|██████████| 391/391 [00:10<00:00, 39.05it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4797, Accuracy: 8359/10000 (83.59%)

[EPOCH 37 / 500] -- 
Loss=0.5111714005470276 Batch_id=390 Accuracy=77.49: 100%|██████████| 391/391 [00:10<00:00, 38.30it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4910, Accuracy: 8317/10000 (83.17%)

[EPOCH 38 / 500] -- 
Loss=0.8684131503105164 Batch_id=390 Accuracy=77.72: 100%|██████████| 391/391 [00:10<00:00, 36.68it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4978, Accuracy: 8302/10000 (83.02%)

[EPOCH 39 / 500] -- 
Loss=0.6628566384315491 Batch_id=390 Accuracy=77.43: 100%|██████████| 391/391 [00:10<00:00, 38.33it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4820, Accuracy: 8366/10000 (83.66%)

[EPOCH 40 / 500] -- 
Loss=0.7396952509880066 Batch_id=390 Accuracy=77.73: 100%|██████████| 391/391 [00:10<00:00, 35.77it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Epoch    41: reducing learning rate of group 0 to 3.0000e-03.

Test set: Average loss: 0.4996, Accuracy: 8306/10000 (83.06%)

[EPOCH 41 / 500] -- 
Loss=0.4796312749385834 Batch_id=390 Accuracy=79.01: 100%|██████████| 391/391 [00:10<00:00, 38.53it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4446, Accuracy: 8493/10000 (84.93%)

[EPOCH 42 / 500] -- 
Loss=0.5841337442398071 Batch_id=390 Accuracy=79.55: 100%|██████████| 391/391 [00:11<00:00, 34.50it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4421, Accuracy: 8503/10000 (85.03%)

[EPOCH 43 / 500] -- 
Loss=0.6707457304000854 Batch_id=390 Accuracy=80.07: 100%|██████████| 391/391 [00:10<00:00, 35.66it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4422, Accuracy: 8486/10000 (84.86%)

[EPOCH 44 / 500] -- 
Loss=0.6058759689331055 Batch_id=390 Accuracy=79.77: 100%|██████████| 391/391 [00:09<00:00, 39.24it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4382, Accuracy: 8509/10000 (85.09%)

[EPOCH 45 / 500] -- 
Loss=0.6127276420593262 Batch_id=390 Accuracy=79.97: 100%|██████████| 391/391 [00:09<00:00, 39.37it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4381, Accuracy: 8507/10000 (85.07%)

[EPOCH 46 / 500] -- 
Loss=0.6135624647140503 Batch_id=390 Accuracy=80.03: 100%|██████████| 391/391 [00:09<00:00, 40.13it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4402, Accuracy: 8493/10000 (84.93%)

[EPOCH 47 / 500] -- 
Loss=0.6020029783248901 Batch_id=390 Accuracy=80.11: 100%|██████████| 391/391 [00:13<00:00, 29.45it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4320, Accuracy: 8519/10000 (85.19%)

[EPOCH 48 / 500] -- 
Loss=0.42090821266174316 Batch_id=390 Accuracy=79.92: 100%|██████████| 391/391 [00:10<00:00, 36.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4281, Accuracy: 8541/10000 (85.41%)

[EPOCH 49 / 500] -- 
Loss=0.6142758727073669 Batch_id=390 Accuracy=80.18: 100%|██████████| 391/391 [00:11<00:00, 33.66it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4377, Accuracy: 8518/10000 (85.18%)

[EPOCH 50 / 500] -- 
Loss=0.6704979538917542 Batch_id=390 Accuracy=80.03: 100%|██████████| 391/391 [00:09<00:00, 41.12it/s] 
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4292, Accuracy: 8522/10000 (85.22%)
```
