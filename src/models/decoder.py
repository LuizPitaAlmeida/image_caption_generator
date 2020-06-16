"""decoder
This module contains the deep learning decoder of a model for image captioning
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
from bert import Bert
from softattention import SoftAttention

# Defines
ENCODER_EMBEDDING_DIM = 2048
BERT_EMBEDDING_DIM = 768
ATTENTION_EMBEDDING_DIM = 512
DECODER_EMBEDDING_DIM = 512
DROPOUT_VALUE = 0.5
DEFAULT_DEVICE = torch.device("cuda:0")


class Decoder(torch.nn.Module):
    """
    Decoder.
    """
    def __init__(self, vocab, encoder_dim=ENCODER_EMBEDDING_DIM,
                 bert_emb_dim=BERT_EMBEDDING_DIM,
                 attention_dim=ATTENTION_EMBEDDING_DIM,
                 decoder_dim=DECODER_EMBEDDING_DIM, dropout=DROPOUT_VALUE,
                 device=DEFAULT_DEVICE):
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.dropout = dropout

        # BERT INIT
        self.bert = Bert(device=device)

        # ATTENTION INIT
        self.attention = SoftAttention(encoder_dim, decoder_dim, attention_dim)

        # LSTM DECODER INIT
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.decode_step = torch.nn.LSTMCell(
            bert_emb_dim + encoder_dim, decoder_dim, bias=True)
        # linear layer to find initial hidden state of LSTMCell
        self.init_hidden = torch.nn.Linear(encoder_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init_cell = torch.nn.Linear(encoder_dim, decoder_dim)
        # linear layer to create a sigmoid-activated gate
        self.f_beta = torch.nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = torch.nn.Sigmoid()

        # WORDS CLASSIFIER INIT
        # linear layer to find scores over vocabulary
        self.fc = torch.nn.Linear(decoder_dim, self.vocab_size)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, encoder_out, captions_ids, caption_lengths):
        # Get info from encoder output and flatten it
        encoder_out, batch_size, encoder_dim, num_pixels = \
            self._encoder_info(encoder_out)

        # We won't decode at the <end> position, since we've finished
        # generating as soon as we generate <end>. So, decoding lengths
        # are actual lengths - 1
        decode_lengths = [cap_len-1 for cap_len in caption_lengths]

        # Init variables
        hidden, cell, predictions, alphas = self._init_variables(
            encoder_out, batch_size, num_pixels, decode_lengths
        )

        bert_emb = self.bert(captions_ids, decode_lengths, self.vocab)

        predictions, alphas = self._loop_for_attention_word_generation(
            encoder_out, bert_emb, decode_lengths, hidden, cell,
            predictions, alphas
        )

        return predictions, captions_ids, decode_lengths, alphas

    def _encoder_info(self, encoder_out):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        return encoder_out, batch_size, encoder_dim, num_pixels

    def _init_variables(self, encoder_out, batch_size, num_pixels,
                        decode_lengths):
        # Init hidden and cell states with encoder mean value
        mean_encoder_out = encoder_out.mean(dim=1)
        hidden = self.init_hidden(mean_encoder_out)
        cell = self.init_cell(mean_encoder_out)

        # Create tensors to hold word prediction scores and
        # alphas
        predictions = torch.zeros(
            batch_size, max(decode_lengths), self.vocab_size
        )
        alphas = torch.zeros(
            batch_size, max(decode_lengths), num_pixels
        )

        return hidden, cell, predictions, alphas

    def _loop_for_attention_word_generation(
        self, encoder_out, bert_emb, decode_lengths, hidden, cell,
        predictions, alphas
    ):
        """At each time-step, decode by attention-weighing the encoder's
        output based on the decoder's previous hidden state output then
        generate a new word in the decoder with the previous word and the
        attention weighted encoding
        """
        for t in range(max(decode_lengths)):
            batch_size_t = sum(
                [length > t for length in decode_lengths])

            # apply soft attention mechanism
            att_weighted_enc, alpha = self.attention(
                    encoder_out[:batch_size_t], hidden[:batch_size_t]
                )

            # gating scalar
            gate = self.sigmoid(self.f_beta(hidden[:batch_size_t]))

            att_weighted_enc = gate * att_weighted_enc

            # Apply LSTM Decoder
            hidden, cell = self.decode_step(torch.cat(
                [bert_emb[:batch_size_t, t, :], att_weighted_enc], dim=1),
                (hidden[:batch_size_t], cell[:batch_size_t]))

            # Classify
            preds = self.fc(self.dropout(hidden))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, alphas
