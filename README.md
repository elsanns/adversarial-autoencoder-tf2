# Adversarial autoencoder TF2
A Tensorflow 2.0 implementation of __[Adversarial Autoencoder](https://arxiv.org/abs/1511.05644/)__ (ICLR 2016)

<br>

## Model
Architecture | Description
------------ | -------------
<img src="imgs/aae-fig3.png" width="800px" style="max-width:100%"> | Regularization of the hidden code by incorporationg full label information (Fig.3 from the paper).<br/> <sub>*Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, and Ian J. Goodfellow. 2015. Adversarial Autoencoders. CoRRabs/1511.05644 (2015). Figure 3 from the paper.*</sub>

<br>

## Results for `gaussian_mixture` prior

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

<br>

## Example of usage
```
python train_model.py --prior_type gaussian_mixture
```

### Attributes
* `--prior_type`: Type of target prior distribution. Default: `gaussian_mixture`. Required.
* `--results_dir`: Training visualization directory. Default: `results`. Created if non-existent.
* `--log_dir`: Log directory (Tensorboard). Default: `logs`. Created if non-existent.
* `--gm_x_stddev`: Gaussian mixture prior: standard deviation for the x coord. Default: `0.5`
* `--gm_y_stddev`: Gaussian mixture prior: standard deviation for the y coord. Default: `0.1`
* `--n_epochs`: Number of epochs. Default: `20`
* `--learning_rate`: Learning rate. Default: `0.001`
* `--batch_size`: Batch size. Default: `128`
* `--num_classes`: Number of classes (for further use). Default: `10`

## Visualization of outliers
Visualization of outliers from learnt distribution in the latent space
<img src="imgs/outliers-all.png" width="1200px" style="max-width:100%"> 
