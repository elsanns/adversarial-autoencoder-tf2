# Adversiarial autoencoder TF2

A Tensorflow 2.0 implementation of __[Adversarial Autoencoder](https://arxiv.org/abs/1511.05644/)__ (ICLR 2016)

## Model
Architecture | Description
------------ | -------------
<img src="imgs/aae-fig3.png" width="800px" style="max-width:100%"> | Regularization of the hidden code by incorporationg full label information (Fig.3 from the paper).<br/> <sub>*Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, and Ian J. Goodfellow. 2015. Adversarial Autoencoders. CoRRabs/1511.05644 (2015). Figure 3 from the paper.*</sub>

## Results
### Latent space
Target prior distribution | Learnt latent space | Sampled decoder ouput
------------ | ------------- |  ------------- 
<img src="imgs/gaussian_mixture_target_prior.png" width="300px" style="max-width:100%"> |<img src="imgs/learnt_manifold_example.png" width="300px" style="max-width:100%">| <img src="imgs/sampled_decoder_output.png" width="200px" style="max-width:100%">

### Reconstruction
Input images | Reconstructed images 
------------ | ------------- 
<img src="imgs/input_images.png" width="200px" style="max-width:100%"> |<img src="imgs/reconstruction_example.png" width="200px" style="max-width:100%">


### Training loss
Gan | Encoder | Discriminator
------------ | ------------- |  -------------
<img src="imgs/gan_loss.png" width="300px" style="max-width:100%"> | <img src="imgs/encoder_loss.png" width="300px" style="max-width:100%"> | <img src="imgs/discriminator_loss.png" width="300px" style="max-width:100%"> 

## Usage
```
python train_model.py --prior_type gaussian_mixture
```
