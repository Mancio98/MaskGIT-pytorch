import torch
import torch.nn as nn


class Codebook(nn.Module):
    """
    Codebook mapping: takes in an encoded image and maps each vector onto its closest codebook vector.
    Metric: mean squared error = (z_e - z_q)**2 = (z_e**2) - (2*z_e*z_q) + (z_q**2)
    """

    def __init__(self, args):
        super().__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta #weight factor for the codebook loss

        # embedding matrix - A simple lookup table that stores embeddings of a fixed dictionary and size.
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        # initialize the weights by a uniform distribution
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        # prepare the latent vector to find the minimum distance between its and codebook vector
        z = z.permute(0, 2, 3, 1).contiguous() # put the channels (1) as last dimension
        z_flattened = z.view(-1, self.latent_dim) # and flatten

        # dist between all latent vec to all codebook vectors - expanded version of L2 (a-b)^2 = a^2 -2ab +b^2
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t()) # Tensor.t() transpose

        # get the min dist indices of codebook foreach latent vec
        min_encoding_indices = torch.argmin(d, dim=1)
        #get actual codebook vectors from indices
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # codebook loss - .detach() is the stop gradient
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients for backward flow
        z_q = z + (z_q - z).detach()  # moving average instead of hard codebook remapping

        # guardare originale (taming)
        # perplexity
        # e_mean = torch.mean(min_encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 3, 1, 2) #move back the channels to orig pos

        #  return z_q, loss, (perplexity, min_encodings, min_encoding_indices)
        return z_q, min_encoding_indices, loss #return indices for the transformer part