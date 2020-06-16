"""softattention
This module contains the deep learning attention mechanism of a model for
image captioning task. This model is in a encoder-decoder architecture. The
encoder is a RESNET101 CNN that extract image features. On the other hand, the
decoder is formed by a LSTM layer to do captions predictions. The LSTM layer
is feed by some image features and a set of word embedding provided by a BERT
model acting over train dataset captions. Each predicted word feed a Soft
Attention layers that applies weights for each pixel in image, saying to the
model where to look. This weighted image embedding is the above-mentioned
image feature that is inputted back into LSTM for next word prediction.

All this code is based on the paper:

Xu, Kelvin, et al. "Show, attend and tell: Neural image caption generation
with visual attention." International conference on machine learning. 2015.

And Github implementations provided in:

- <https://github.com/ajamjoom/Image-Captions>
- <https://github.com/yunjey/pytorch-tutorial>
- <https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning>

Thanks Abdulrahman Jamjoom, Yunjey Choi, and Sagar Vinodababu
"""

import torch

# DEFINES
ENCODER_IMG_EMBEDDING_SIZE = 2048
DECODER_HIDDEN_DIM = 512
ATTENTION_EMB_DIM = 512


class SoftAttention(torch.nn.Module):
    '''Soft Attention defined in Show, Attend and Tell paper. It consists of a
    series of linear layers that merges hidden states from decoder and image
    features from encoder. After, this linear merge is classified using a
    softmax to determine the pixels that are more relevant to predict the next
    word.
    '''
    def __init__(self, encoder_dim=ENCODER_IMG_EMBEDDING_SIZE,
                 decoder_hidden_dim=DECODER_HIDDEN_DIM,
                 attention_dim=ATTENTION_EMB_DIM):
        """The constructor initializes the soft Attention layers.

        Parameters:
        - encoder_dim (int): Output dimension of image of features embeddings
        - decoder_hidden_dim (int): Dimension of decoder hidden cells
        - attention_dim (int): Input size of common attention layer.
        """
        super(SoftAttention, self).__init__()
        self.encoder_layer = torch.nn.Linear(encoder_dim, attention_dim)
        self.decoder_layer = torch.nn.Linear(decoder_hidden_dim, attention_dim)
        self.common_attention = torch.nn.Linear(attention_dim, 1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """Torch Model Main function

        Input
        - encoder_out: [batch_size, n_pixels, encoder_dim]
        - decoder_hidden: [batch_size, decoder_hidden_dim]

        Output
        - encoder_layer: [batch_size, n_pixels, attention_dim]
        - decoder_layer: [batch_size, attention_dim]
        - relu: [batch_size, n_pixels, attention_dim]
        - common_attention with squeeze(2): [batch_size, n_pixels]
        - softmax: [batch_size, n_pixels]
        - encoder_out_with_attention: [batch_size, encoder_dim]

        Notes
        - n_pixels = emb_img_size*emb_img_size
        """
        enc_attention = self.encoder_layer(encoder_out)
        dec_attention = self.decoder_layer(decoder_hidden)
        # Pass a ReLu over the addition of encoder e decoder attention
        # embeddings
        activation = self.relu(enc_attention + dec_attention.unsqueeze(1))
        attention_emb = self.common_attention(activation).squeeze(2)
        alpha = self.softmax(attention_emb)

        # Apply attention in encoded images. It does the product between alpha
        # and encoder_out, finally flatten the result summing the pixels values
        encoder_out_with_attention = \
            (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return encoder_out_with_attention, alpha
