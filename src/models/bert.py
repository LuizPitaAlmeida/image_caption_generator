"""bert
This module contains the deep learning BERT transformer model of a model for
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

More about Bert model refer to its paper:

Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers
for language understanding." arXiv preprint arXiv:1810.04805 (2018).
"""

import torch
from transformers import BertTokenizer, BertModel

# DEFINES
CLS_TOKEN = u'[CLS] '


class Bert(torch.nn.Module):
    """
    Bert model to generate word embeddings features for decoder step.
    Bert uses the input captioning info to generate a new representation
    of words associated to each image. The Bert output together with
    image representation (after apply soft attention) feed a LSTM layer
    that will predict a new caption for the image.

    Notice that BERT is also an attention model, meaning that both images
    and captions embeddings are formed by attention mechanisms.
    """
    def __init__(self):
        """The constructor initializes the Bert layer, its tokenizer and
        freeze its parameters to avoid train it.
        """
        super(Bert, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

        self.device = bert.weight.device
        print(self.device)

    def forward(self, captions_ids, decode_lengths, vocab, pad_id=0):
        """ Predict a BERT embedding for each caption in captions_ids.

        Input
        - captions_ids: [batch_size, sequence_length]

        Output
        - stacked embeddings: [batch_size, limited_sequence+2, bert_emb_dim]

        Note
        - sequence length is limited by max(decode_lengths)
        - the +2 value in dimension 1 of stacked embeddings is due to the
          adition of ['CLS'] token and <end> token which was desconsidered in
          decode_lengths.

        Default values (BERT parameters)
        - bert_emb_dim = 768
        """
        embeddings = []
        max_dec_len = max(decode_lengths)

        # loop over each image caption
        for caption_ids in captions_ids:

            # padding caption to get fixed length size
            while len(caption_ids) < max_dec_len:
                caption_ids.append(pad_id)

            # get back ids to words to apply bert tokenization
            caption, tokenized_cap, indexed_tokens = \
                self._redo_tokenization(caption_ids, vocab)

            # Predict embeddings
            bert_predictions = self._predict(indexed_tokens)

            # Associate the portions of BERT embeddings to caption
            # and its tokens
            tokens_embedding = self._pred2tokens_embeddings(
                caption, tokenized_cap, bert_predictions
            )

            embeddings.append(torch.stack(tokens_embedding))
        return torch.stack(embeddings)

    def _redo_tokenization(self, cap_id, vocab):
        caption = ' '.join([vocab.idx2word[word_idx.item()]
                            for word_idx in cap_id])
        caption = CLS_TOKEN + caption
        tokenized_cap = self.tokenizer.tokenize(caption)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_cap)
        return caption, tokenized_cap, indexed_tokens

    def _predict(self, tokens_id):
        bert_embedding, _ = self.bert(
            torch.tensor([tokens_id]).to(self.device)
        )
        return bert_embedding.squeeze(0)

    def _pred2tokens_embeddings(self, caption, tokenized, predicted):
        """Walk throw BERT tokenized captions (tokenized) concatenating tokens
        until it forms a full token presente in caption. When it occurs the
        embeddings in predicted, that corresponds to each concatenated token in
        tokenized, are appended to tokens_embedding.
        """
        tokens_embedding = []

        # split captions to get all full tokens
        splitted_caption = caption.split()

        # Caption walker variable. It will be incremented when the tokens
        # from tokenized form a full token of caption
        caption_walker = 0

        for cap_full_token in splitted_caption:
            # keep the value of tokenized current token been compared with
            # caption current full token
            current_token = ''
            # variable used to count the number of tokens in tokenized
            # necessary to compound a token in caption (full token)
            n_tokens_concatenated = 0

            # Walk trow tokenized concatenating tokens and comparing the
            # result with full token. We start walk from index 1 to disregard
            # CLS token
            for tokenized_walker, _ in enumerate(tokenized[1:]):
                token = tokenized[tokenized_walker+caption_walker]
                piece_embedding = predicted[tokenized_walker+caption_walker]

                # When token in tokenize is a full token
                if token == cap_full_token and current_token == '':
                    # append the embeddings of this token
                    tokens_embedding.append(piece_embedding)
                    caption_walker += 1
                    break  # condition satisfied
                # When token in tokenize is a partial token
                else:
                    n_tokens_concatenated += 1
                    # When found a partial token for the first time
                    if current_token == '':
                        # Append the initial embedding of this token
                        tokens_embedding.append(piece_embedding)
                        # Use current_token to keep the partial token,
                        # replacing the partial token Bert symbol (#)
                        current_token += token.replace('#', '')
                    # Next interation over partial tokens
                    else:
                        # increment the initial embedding saved
                        tokens_embedding[-1] = torch.add(
                            tokens_embedding[-1], piece_embedding)
                        # Update current token adding another partial token
                        current_token += token.replace('#', '')

                        # Check if the concatenated partial tokens formed
                        # the full token
                        if current_token == cap_full_token:
                            # caption_walker is incremented with the number
                            # of tokens used to form the full token
                            caption_walker += n_tokens_concatenated
                            break
        return tokens_embedding
