# CNN Architectures

Compilation of some convolutional neural network architectures for binary object segmentation. The networks currently implemented are:
- U-Net, formulated in *U-Net: Convolutional Networks for Biomedical Image Segmentation*, by O. Ronneberger, P. Fisher and T. Brox, in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015.
- GSC, formulated in *Promising crack segmentation method based on gated skip connection*, by M. Jabreel and M. Abdel-Nasser, in Electronic Letters, Volume 56, Pages 493-495, 2020.
- DoubleU-Net, formulated in *DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation*, by D. Jha, M. Riegler, D. Johansen, P. Halvorsen, H. Johansen, in the IEEE 33rd International Symposium on Computer-Based Medical Systems (CBMS), 2020.
- ESNet, formulated in *ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation*, by Yu Wang, Quan Zhou and Xiaofu Wu, in Chinese Conference on Pattern Recognition and Computer Vision, 2019.
- ERFNet, formulated in *ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation*, by E. Romera, J. M. Álvarez, L. M. Bergasa and R. Arroyo, in IEEE Transactions on Intelligent Transportation Systems, vol. 19, no. 1, pp. 263-272, 2018.
- CGNet, formulated in *CGNet: A Light-weight Context Guided Network for Semantic Segmentation*, by T. Wu, S. Tang, R. Zhang, J. Cao and Y. Zhang, in IEEE Transactions on Image Processing, vol. 30, pp. 1169-1179, 2021.
- LinkNet, formulated in *LinkNet: Exploiting Encoder Representations for
Efficient Semantic Segmentation*, by A. Chaurasia and E. Culurciello, in IEEE Visual Communications and Image Processing (VCIP), 2017.
- Pretrained UNet (UNetPre), formulated in *Automatic Lung Segmentation in Chest X-ray Images Using Improved U-Net*, by Liu, Wufeng & Luo, Jiaxin & Yang, Yan & Wang, Wenlian & Deng, Junkui & Yu, Liang, 2022

In addition, the U-NetUIB architecture has been implemented. It has the same implementation as U-Net, but with the particularity that it allows a binary classification of the image from the segmentation. The main motivation of this implementation is the classification of post-surgical wound images according to the presence or absence of infection in the wound, based on wound segmentation.

## Acknowledgements

The original code for the networks was implemented by the following people:
- U-Net, by [Miquel Miró Nicolau](https://github.com/miquelmn), PhD student on the Balearic Islands University.
- GSC, DoubleU-Net, ESNet, ERFNet, CGNet, LinkNet and UNetPre, by [Antonio Nadal Martínez](https://github.com/nmantonio), mathematics undergraduate student on the Balearic Islands University.

The code that appears in this package has been modified to unify the notation and structure.

## Installation

The package is slated to become available on the PyPI package manager shortly. In the interim, it can be installed using `pip` with the following command:

```
pip install git+https://github.com/mmunar97/cnn_architectures.git@new_architectures
```
