from torch.nn.utils.rnn import pack_padded_sequence
import skimage.transform
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


IMG_EXHIBITION_DIM = 224
ATTENTION_PLOT_COLS = 5


def print_sample(vocab, hypotheses, references, images, sample_id, alphas,
                 show_att):

    hyp_sentence = [vocab.idx2word[word_idx] for word_idx
                    in hypotheses[sample_id]]

    ref_sentence = [vocab.idx2word[word_idx] for word_idx
                    in references[sample_id]]

    image = images[0][sample_id]
    image = np.uint8(255*(image - np.min(image))/np.ptp(image))
    image = Image.fromarray(image, 'RGB')
    image = image.resize([IMG_EXHIBITION_DIM, IMG_EXHIBITION_DIM])

    print('Hypotheses: '+" ".join(hyp_sentence))
    print('References: '+" ".join(ref_sentence))

    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    if show_att:
        plt.figure(figsize=(15, 15))
        for pos, word in enumerate(hyp_sentence):
            plt.subplot(
                np.ceil(len(hyp_sentence) / ATTENTION_PLOT_COLS),
                ATTENTION_PLOT_COLS, pos+1
            )
            plt.text(0, 1, '%s' % (word), color='black',
                     backgroundcolor='white', fontsize=12)
            plt.imshow(image, cmap='gray')

            current_alpha = alphas[0][pos, :].detach().numpy()
            alpha = skimage.transform.resize(
                current_alpha, [IMG_EXHIBITION_DIM, IMG_EXHIBITION_DIM]
            )
            if pos == 0:
                plt.imshow(alpha, alpha=0, cmap='gray')
            else:
                plt.imshow(alpha, alpha=0.7, cmap='gray')
            plt.axis('off')
        plt.show()
