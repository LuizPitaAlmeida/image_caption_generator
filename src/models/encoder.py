"""encoder
This module contains the deep learning encoder of a model for image captioning
task. This model is in a encoder-decoder architecture. The encoder is a
RESNET101 CNN that extract image features. On the other hand, the decoder is
formed by a LSTM layer to do captions predictions. The LSTM layer is feed by
some image features and a set of word embedding provided by a BERT model
acting over train dataset captions. Each predicted word feed a Soft Attention
layers that applies weights for each pixel in image, saying to the model where
to look. This weighted image embedding is the above-mentioned image feature
that is inputted back into LSTM for next word prediction.

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
import torchvision

# DEFAULT VALUES
EMB_IMG_SIZE = 14


# ENCODER
class Encoder(torch.nn.Module):
    """Encoder class transforms an image into an embedding representation. Due
    the fact that we remove the classification layers of used Resnet101 CNN
    model, this module could be consider a image feature extraction approach.
    """
    def __init__(self, emb_img_size=EMB_IMG_SIZE, finetune=False):
        """The constructor initializes the ResNet101 CNN model and the encoder
        layer.

        Parameters:
        - emb_img_dim (int): Output dimension of image of features embeddings
        - finetune (bool): If True, do a finetunning on CNN layers, but keeping
                           initial convolutional layers with original weights
        """
        super(Encoder, self).__init__()

        # Load an pretrained ResNet101 model
        resnet101 = torchvision.models.resnet101(pretrained=True)
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet101.children())[:-2]
        self.cnn_model = torch.nn.Sequential(*modules)
        # Freeze model weights to not train
        for param in self.cnn_model.parameters():
            param.requires_grad = False
        # If fine-tuning, only fine-tunes convolutional blocks 2 through 4
        if finetune:
            for convolutional_layer in list(self.cnn_model.children())[5:]:
                for param in convolutional_layer.parameters():
                    param.requires_grad = True

        # Encoder layer that generates image features embedding vectors of
        # shape:
        # [batch_size, resnet101_embedding_dim, emb_img_dim, emb_img_dim]
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d(
            (emb_img_size, emb_img_size)
        )

    def forward(self, image):
        """Torch Model Main function

        Input
        - image: [batch_size, n_channels, img_rows, img_cols]

        Output
        - cnn_model: [batch_size, emb_dim, last_layer_dim, last_layer_dim]
        - adaptive_pool: [batch_size, emb_dim, emb_img_size, emb_img_size]
        - permute: [batch_size, emb_img_size, emb_img_size, emb_dim]

        Default values (Defined by resnet101 model)
        - img_rows = img_cols = 244
        - n_channels = 3
        - last_layer_dim = 7
        - emb_dim = 2048
        """
        out = self.cnn_model(image)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out
