# CNN Architectures

Compilation of some convolutional neural network architectures for binary object segmentation. The networks currently implemented are:
- U-Net, formulated in *U-Net: Convolutional Networks for Biomedical Image Segmentation*, by O. Ronneberger, P. Fisher and T. Brox, in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015.
- GSC, formulated in *Promising crack segmentation method based on gated skip connection*, by M. Jabreel and M. Abdel-Nasser, in Electronic Letters, Volume 56, Pages 493-495, 2020.
- DoubleU-Net, formulated in *DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation*, by D. Jha, M. Riegler, D. Johansen, P. Halvorsen, H. Johansen, in the IEEE 33rd International Symposium on Computer-Based Medical Systems (CBMS), 2020.

In addition, the U-NetUIB architecture has been implemented. It has the same implementation as U-Net, but with the particularity that it allows a binary classification of the image from the segmentation. The main motivation of this implementation is the classification of post-surgical wound images according to the presence or absence of infection in the wound, based on wound segmentation.

## Acknowledgements

The original code for the networks was implemented by the following people:
- U-Net, by [Miquel Miró Nicolau](https://github.com/miquelmn), PhD student on the Balearic Islands University.
- GSC and DoubleU-Net, by Antonio Nadal Martínez, mathematics undergraduate student on the Balearic Islands University.