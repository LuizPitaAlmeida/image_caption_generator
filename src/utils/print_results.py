import torch
from torch.nn.utils.rnn import pack_padded_sequence
import skimage.transform
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

PAD = 0
START = 1
END = 2
UNK = 3
IMG_EXHIBITION_DIM = 224
ATTENTION_PLOT_COLS = 5


def compute_sample_results(batch_sample, decoder_out):
    references = []
    test_references = []
    hypotheses = []
    all_imgs = []
    all_alphas = []

    num_batches = len(batch_sample)
    for i, inp in enumerate([batch_sample]):
        imgs, caps, caplens = inp
        if i > 0:
            break

        # Get scores and targets
        scores, caps_sorted, decode_lengths, alphas = decoder_out
        targets = caps_sorted[:, 1:]
        # Remove timesteps that we didn't decode at, or are pads
        scores_packed = pack_padded_sequence(
            scores, decode_lengths, batch_first=True)[0]
        targets_packed = pack_padded_sequence(
            targets, decode_lengths, batch_first=True)[0]

        # References
        for j in range(targets.shape[0]):
            # validation dataset only has 1 unique caption per img
            img_caps = targets[j].tolist()
            # remove pad, start, and end
            clean_cap = [w for w in img_caps if w not in [PAD, START, END]]
            img_captions = list(map(lambda c: clean_cap, img_caps))
            test_references.append(clean_cap)
            references.append(img_captions)

        # Hypotheses
        _, preds = torch.max(scores, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            pred = p[:decode_lengths[j]]
            pred = [w for w in pred if w not in [PAD, START, END]]
            temp_preds.append(pred)  # remove pads, start, and end
        preds = temp_preds
        hypotheses.extend(preds)

        # Images
        imgs_jpg = imgs.numpy()
        imgs_jpg = np.swapaxes(np.swapaxes(imgs_jpg, 1, 3), 1, 2)
        all_alphas.append(alphas)
        all_imgs.append(imgs_jpg)

        return hypotheses, test_references, all_imgs, all_alphas


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
    else:
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.show()
