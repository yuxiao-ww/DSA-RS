import torch
import torch.nn as nn
import torch.nn.functional as F


class TextAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dims):
        super(TextAutoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.ModuleList()
        for i, (in_features, out_features) in enumerate(zip([input_dim] + encoding_dims[:-1], encoding_dims)):
            self.encoder.append(nn.Linear(in_features, out_features))

        # Decoder layers - reverse of encoder layers
        self.decoder = nn.ModuleList()
        reversed_dims = list(reversed(encoding_dims))
        for i, (in_features, out_features) in enumerate(zip(reversed_dims, reversed_dims[1:] + [input_dim])):
            self.decoder.append(nn.Linear(in_features, out_features))

    def forward(self, x):
        # Encode
        for layer in self.encoder:
            x = F.relu(layer(x))
        z = x  # Encoded representation
        # Decode
        for layer in self.decoder:
            x = F.relu(layer(x))
        reconstructed = torch.sigmoid(x)  # Assuming input data is normalized between [0, 1]
        return z, reconstructed

