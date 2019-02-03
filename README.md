# NISE_project GAN
Neuro-Inspired System Engineering - final project. Tetiana Klymenko and Cristina Gil
Files for model of cGAN and its training.

ipynb notebooks (for training the models):
train_Linear_cGAN - training linear model of cGAN
NISE GanConv - training convolutional model for cGAN and evaluating KL divergence

Models (classes for models and Data Loader):
SignDataLoad.py - Dataloader for Sign Language MNIST data base (cannot be stored on github due to the memory limitations)
ganSigns.py - model for linear cGAN
ganConv.py - model for convolutional cGAN

Trained (contains .pth files with trained models)
discriminator_conv_short.pth - descriminator for convolutional cGAN pretrained on 6 classes.
generator_conv_short.pth - generator for convolutional cGAN pretrained on 6 classes.



