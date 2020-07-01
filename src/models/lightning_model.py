"""lightning_model
The Pytorch Lightning is a software package over Pytorch that tries to free
scientists of software engineering's problems. It is used to organize Pytorch
codes in a simple style giving to scientist a easily way to modeling, train,
valid and test deep learning models.
Please, read more in (<https://github.com/PyTorchLightning/pytorch-lightning>)

This code refers to Pytorch lightning model module. Which contains the train
and valid algorithm of the Encoder-Decoder model for image captioning
generation.

It was designed to run in a GPU.
"""

import numpy as np

import torch
from torch.nn.utils.rnn import pack_padded_sequence
import pytorch_lightning as pl
from nltk.translate.bleu_score import corpus_bleu

from utils.hardware_stats import HardwareStats
from .decoder import Decoder
from .encoder import Encoder


class ImageCaptioning(pl.LightningModule):
    """Encoder-Decoder Image Caption Model"""
    def __init__(self, train_loader, valid_loader, vocab):
        super().__init__()

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.encoder = Encoder().cuda()
        self.decoder = Decoder(vocab=vocab).cuda()
        self.loss = torch.nn.CrossEntropyLoss()

        # Keep validation results
        self.references = []
        self.hypotheses = []

        self.PAD = 0
        self.START = 1
        self.END = 2
        self.UNK = 3

        self.hardware_stats = HardwareStats()

    def forward(self, x):
        images, captions_ids, captions_len = x
        encoder_out = self.encoder(images.cuda())
        outputs = self.decoder(
            encoder_out, captions_ids.cuda(), captions_len)
        return outputs

    def training_step(self, x, batch_idx):
        predictions, captions_id, decoded_lengths, alphas = self(x)
        scores = pack_padded_sequence(
            predictions, decoded_lengths, batch_first=True).data
        y = captions_id[:, 1:]
        y = pack_padded_sequence(y, decoded_lengths, batch_first=True)[0]

        loss = self.loss(scores.cuda(), y.cuda())
        loss += ((1. - alphas.cuda().sum(dim=1)) ** 2).mean()
        loss.cuda()

        return {"loss": loss, "log": {"loss": loss},
                "progress_bar": self.hardware_stats.hardware_stats()}

    def validation_step(self, x, batch_idx):
        predictions, captions_id, decoded_lengths, alphas = self(x)
        scores = pack_padded_sequence(
            predictions, decoded_lengths, batch_first=True).data

        # Calc loss
        y = captions_id[:, 1:]
        y = pack_padded_sequence(y, decoded_lengths, batch_first=True).data
        loss = self.loss(scores.cuda(), y.cuda())
        loss += (((1. - alphas.cuda().sum(dim=1)) ** 2).mean())
        loss.cuda()

        # References
        original_caption = []
        targets = captions_id[:, 1:]
        for j in range(targets.shape[0]):
            # validation dataset only has 1 unique caption per img
            img_caps = targets[j].tolist()
            # remove pad, start, and end
            clean_cap = [w for w in img_caps
                         if w not in [self.PAD, self.START, self.END]]
            img_caption = list(map(lambda c: clean_cap, img_caps))
            self.references.append(img_caption)
            original_caption.append(clean_cap)

        # Hypotheses
        _, preds = torch.max(predictions, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            pred = p[:decoded_lengths[j]]
            pred = [w for w in pred
                    if w not in [self.PAD, self.START, self.END]]
            temp_preds.append(pred)
        preds = temp_preds
        self.hypotheses.extend(preds)

        progress_bar = self.hardware_stats.hardware_stats()
        progress_bar.update({"loss": loss})

        return {"loss": loss, "preds": preds, "img_captions": original_caption,
                "progress_bar": progress_bar}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["batch_loss"] for x in outputs]).mean()

        return {"log": {"train_loss": avg_loss}}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        references = [x["img_captions"] for x in outputs]
        hypotheses = [x["preds"] for x in outputs]

        bleu = corpus_bleu(self.references, self.hypotheses)
        bleu_1 = corpus_bleu(self.references, self.hypotheses,
                             weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(self.references, self.hypotheses,
                             weights=(0, 1, 0, 0))
        bleu_3 = corpus_bleu(self.references, self.hypotheses,
                             weights=(0, 0, 1, 0))
        bleu_4 = corpus_bleu(self.references, self.hypotheses,
                             weights=(0, 0, 0, 1))

        print("Validation loss:", val_loss)
        print("BLEU:", bleu)
        print("BLEU-1:", bleu_1)
        print("BLEU-2:", bleu_2)
        print("BLEU-3:", bleu_3)
        print("BLEU-4:", bleu_4)

        tensorboard_logs = {"val_loss": val_loss, "bleu": bleu,
                            "bleu_1": bleu_1, "bleu_2": bleu_2,
                            "bleu_3": bleu_3, "bleu_4": bleu_4}

        torch.cuda.empty_cache()
        self.references.clear()
        self.hypotheses.clear()

        return {"val_loss": val_loss, "bleu": bleu, "bleu_1": bleu_1,
                "bleu_2": bleu_2, "bleu_3": bleu_3, "bleu_4": bleu_4,
                "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader
