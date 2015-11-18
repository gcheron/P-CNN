Information
===========

This package contains a matlab implementation of Pose-based CNN (P-CNN) algorithm described in \[1\]. It includes pre-trained CNN appearance vgg-f model \[2\], a matlab version of the flow model of \[3\] and the optical flow implementation of \[4\]. CNN implementation uses the MatConvNet library \[5\]. The project webpage is http://www.di.ens.fr/willow/research/p-cnn/ .

####To run this package:
- Prepare/download CNN models and data examples by running `init.sh` file from the P-CNN folder.
- This package compiles MatConvNet \[5\] in "CPU mode". To speed up computation you may want to enable GPU support (much faster). To help you, we provide the `my_build.m` file in the `matconvnet-1.0-beta11` folder that you can modify.
- You may want to recompile Brox optical flow 2004 \[4\] ([download sources](http://lmb.informatik.uni-freiburg.de/resources/software.php)).

####demo.m
An example of P-CNN computation is given in this package. It computes P-CNN for few videos of the JHMDB dataset \[6\] (for 2 different splits) using pose ground truth annotations.
The `reproduce_ICCV15_results` command reproduces the P-CNN results reported in \[1\]. Because we wanted to provide a "full matlab code", we converted all the code to matlab resulting to a slightly different result (-0.9% accuracy) from the published version due to the switch of the CNN package and retraining.

The provided algorithm takes as input the frames of a video and their corresponding pose joints (from ground truth annotation or from your favorite pose detector). There is a `demo.m` file in the package that you should be able to run.

####Cite
If you use this package, please cite:

>@inproceedings{cheronICCV15,<br>
TITLE = {{P-CNN: Pose-based CNN Features for Action Recognition}},<br>
AUTHOR = {Ch{\'e}ron, Guilhem and Laptev, Ivan and Schmid, Cordelia},<br>
BOOKTITLE = {ICCV},<br>
YEAR = {2015},<br>
}

####References
\[1\] G. Chéron, I. Laptev, C. Schmid. P-CNN: Pose-based CNN Features for Action Recognition. ICCV 2015.

\[2\] K. Chatfield, K. Simonyan, A. Vedaldi, and A. Zisserman. Return of the devil in the details: Delving deep into convolutional nets. BMVC 2014.

\[3\] G. Gkioxari and J. Malik. Finding action tubes. CVPR 2015. ACM 2015.

\[4\] T. Brox, A. Bruhn, N. Papenberg, and J. Weickert. High accuracy optical flow estimation based on a theory for warping. ECCV 2004.

\[5\] A. Vedaldi and K. Lenc. MatConvNet - Convolutional Neural Networks for MATLAB. 

\[6\] H. Jhuang, J. Gall, S. Zuffi, C. Schmid, and M. J. Black. Towards understanding action recognition. ICCV 2013.

####Acknowledgements
We graciously thank the authors of the previous code releases and video benchmark for making them publicly available.
