import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook


class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args).to(device=args.device)
        self.decoder = Decoder(args).to(device=args.device)
        self.codebook = Codebook(args).to(device=args.device)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quantized_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        quantized_codebook_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(quantized_codebook_mapping)

        return decoded_images, codebook_indices, q_loss

    # used by transformer later 
    def encode(self, x):
        encoded_images = self.encoder(x) #encode the img
        quantized_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    # used by transformer later 
    def decode(self, z):
        quantized_codebook_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(quantized_codebook_mapping)
        return decoded_images

    # lambda factor to weight the VQ_VAE loss and GAN loss (see the paper)
    def calculate_lambda(self, nll_loss, gan_loss):
        #nll_loss = perceptual reconstruction loss
        #gan_loss = gan_loss

        #its calculate on last layers
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight

        # torch.autograd.grad calculate the gradient, retain_graph means we want to keep the curren
        # computational graph which is just important that the backward pass can function properly
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0] 
        g_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        # λ is a ratio
        λ = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach() # clipping the values to be between 0 and 10k
        return 0.8 * λ

    """
    to start the discriminator later in training, ensure that training doesn't fail to converge.
    In this way the generator has time to learn already something about reconstructing images before
    the dicriminator joins (the training). Otherwise, the discriminator would have an easy time predicting which imgs
    are real and fake, if it joins right from the start 
    disc_factor -> additional weight which it will weight the discriminator.
    Here is set to 1, but it can be choosen even with schedules.
    """
    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    # used for transformer part
    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
        print("Loaded Checkpoint for VQGAN....")

