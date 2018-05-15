# MWCNN
## Abstract
The tradeoff between receptive field size and efficiency is a crucial issue in low level vision. Plain convolutional networks (CNNs) generally enlarge the receptive field at the expense of computational cost. Recently, dilated filtering has been adopted to address this issue. But it suffers from gridding effect, and the resulting receptive field is only a sparse sampling of input image with checkerboard patterns. In this paper, we present a novel multi-level wavelet CNN (MWCNN) model for better tradeoff between receptive field size and computational efficiency. With the modified U-Net architecture, wavelet transform is introduced to re- duce the size of feature maps in the contracting subnetwork. Furthermore, another convolutional layer is further used to decrease the channels of feature maps. In the expanding subnetwork, inverse wavelet transform is then deployed to reconstruct the high resolution feature maps. Our MWCNN can also be explained as the generalization of dilated fil- tering and subsampling, and can be applied to many image restoration tasks. The experimental results clearly show the effectiveness of MWCNN for image denoising, single image super-resolution, and JPEG image artifacts removal.

## Requirements
MatConvNet

## Results


