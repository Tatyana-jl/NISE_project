# NISE_project GAN
Neuro-Inspired System Engineering - final project. <br />
Tetiana Klymenko and Cristina Gil <br />

ipynb notebooks (for training the models): <br />

Full Model - performance of final model <br />
Train FeedForwardNet - training feedforward Network <br /> 
train_Linear_cGAN - training linear model of cGAN <br />
NISE GanConv - training convolutional model for cGAN and evaluating KL divergence <br />

Models (classes for models and Data Loader): <br />
SignDataLoad.py - Dataloader for Sign Language MNIST data base (cannot be stored on github due to the memory limitations) <br />
ganSigns.py - model for linear cGAN <br />
ganConv.py - model for convolutional cGAN <br />
TwoLayerNet.py - model for feedforward network <br />
Solver.py - class to train the two-layer feedforward network <br />

Hand_model.py - class to generate a model of the hand <br />

Trained (contains .pth files with trained models) <br />
discriminator_conv_short.pth - descriminator for convolutional cGAN pretrained on 6 classes. <br />
generator_conv_short.pth - generator for convolutional cGAN pretrained on 6 classes. <br />
twoLayer.pth - feedforward network for hand position classification (trained for variace 15). <br />



