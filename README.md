# Bachelor-Thesis
My Bachelor Thesis: Colorizing grayscale images with Deep Learning techniques.

There were two main approaches, a Conditional GAN (similar to Pix2pix arhcitecture) and an autoencoder. Both had many variations - there was a total of 19 models trained and examined. For the discriminator of the cGAN I used an architecture which, to my knowledge, has not been used before (I might perfectly be wrong though). Some results have small artifacts, but some of them are pretty neat, especially if they go through a simple postprocessing phase.

I trained on portraits and landscapes separately. I used both 30k and 80k datasets.

These are some of the best results obtained:
![licenta](https://user-images.githubusercontent.com/48358732/112650672-4ae74680-8e54-11eb-8178-7a12d3931b59.PNG)
