# NISE_project GAN
Neuro-Inspired System Engineering - final project. <br />
Tetiana Klymenko and Cristina Gil <br />

ipynb notebooks (for training the models): <br />
train_Linear_cGAN - training linear model of cGAN <br />
NISE GanConv - training convolutional model for cGAN and evaluating KL divergence <br />

Models (classes for models and Data Loader): <br />
SignDataLoad.py - Dataloader for Sign Language MNIST data base (cannot be stored on github due to the memory limitations) <br />
ganSigns.py - model for linear cGAN <br />
ganConv.py - model for convolutional cGAN <br />

Trained (contains .pth files with trained models) <br />
discriminator_conv_short.pth - descriminator for convolutional cGAN pretrained on 6 classes. <br />
generator_conv_short.pth - generator for convolutional cGAN pretrained on 6 classes. <br />



