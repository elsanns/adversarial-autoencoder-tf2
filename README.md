# Adversiarial autoencoder TF2

A Tensorflow 2.0 implementation of __[Adversarial Autoencoder](https://arxiv.org/abs/1511.05644/)__ (ICLR 2016)
* Adversarial autoencoder - hidden code regularization<br/><br/>

## Model
Architecture | Description
------------ | -------------
<img src="imgs/aae-fig3.png" width="400px" style="max-width:100%"> | Regularization of the hidden code by incorporationg full label information. Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, and Ian J. Goodfellow. 2015.Adversarial Autoencoders.CoRRabs/1511.05644 (2015). Figure 3 from the paper.

## Results
Target prior distribution | Learnt latent space | Sampled decoder ouput
------------ | ------------- |  ------------- 
<img src="imgs/gaussian_mixture_target_prior.png" width="300px" style="max-width:100%"> |<img src="imgs/learnt_manifold_example.png" width="300px" style="max-width:100%">| <img src="imgs/sampled_decoder_output.png" width="200px" style="max-width:100%">

### Training loss
Gan | Encoder | Discriminator
------------ | ------------- |  -------------
<img src="imgs/gan_loss.png" width="250px" style="max-width:100%"> | <img src="imgs/encoder_loss.png" width="250px" style="max-width:100%"> | <img src="imgs/discriminator_loss.png" width="250px" style="max-width:100%"> 

## Usage
```
train_model.py --prior_type gaussian_mixture
```
