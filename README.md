# Fully Convolutional Networks for Semantic Segmentation

## # Summary
1. fully connected layers -> 1x1 convolution layers
2. end-to-end learning

## # Difference from Paper
1. with Batch Normalization
1. Skip Architecture -> UNet

## # Results 
| Model  | meanIU | Inference time (with GTX 1050) |
| ------------- | ------------- | ------------- |
| (self) FCN-UNet | 86% | 13ms |

## # Samples
### iter = 1000 (RGB image, GT image, Prediction image)
![result](./results/1000_1.jpg)
![result](./results/1000_2.jpg)
![result](./results/1000_3.jpg)
![result](./results/1000_4.jpg)

### iter = 50000 (RGB image, GT image, Prediction image)
![result](./results/50000_1.jpg)
![result](./results/50000_2.jpg)
![result](./results/50000_3.jpg)
![result](./results/50000_4.jpg)

## # Reference
- Fully Convolutional Networks for Semantic Segmentation
- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- U-Net: Convolutional Networks for Biomedical Image Segmentation