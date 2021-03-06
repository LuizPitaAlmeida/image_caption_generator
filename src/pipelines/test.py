"""test
Compute metrics to valid dataset
"""

import torch
from pytorch_lightning import Trainer

from lightning_model import ImageCaptioning
from build_vocab import build_vocab


class TestPrediction():
    def __init__(self, train_dataloader, test_dataloader, vocab, ckpt_path):
        super().__init__()

        self.model = ImageCaptioning(
            train_dataloader, test_dataloader, vocab).cuda()
        self.ckpt_path = ckpt_path
        self.test_data = test_dataloader

    def notebook_test(self):
        trainer = Trainer(resume_from_checkpoint=self.ckpt_path,
                          gpus=1)
        trainer.test(self.model)

    def predict_in_single_batch(self, single_batch=None):
        checkpoint = torch.load(self.ckpt_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if single_batch is None:
            single_batch = next(iter(self.test_data))
        imgs, captions, lengths = single_batch
        out = self.model((imgs.cuda(), captions.cuda(), lengths))
        return out
