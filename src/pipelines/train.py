"""train
Module to start train the pytorch lightining model.
Using this module you can start a new train or restart a
stopped one from a checkpoint file.
"""

import os
import sys
import time
import argparse
import multiprocessing

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from lightining_model import ImageCaptioning
from build_vocab import build_vocab
from image_tranform import transform_function
from data_loader import get_loader

EARLY_STOP = False
BATCH_SIZE = 64
MAX_EPOCHS = 10
ACC_GRAD_BATCHES = 10  # accumulate_grad_batches
VALIDATION_CHECK = 0.5  # val_check_interval
PROGESS_BAR_RATE = 1  # progress_bar_refresh_rate
LIMIT_VALID_BATCHES = 0.01
LIMIT_TRAIN_BATCHES = 0.05

TORCH_CHECKPOINT_EXT = '.ckpt'
TORCH_STATE_DICT_EXT = '.pt'


def parse_args(argv):
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('exp_name', type=str,
                        help='id of experiment. Ex: exp0')
    parser.add_argument('train_img', type=str,
                        help='input training images path')
    parser.add_argument('valid_img', type=str,
                        help='input validation images path')
    parser.add_argument('annotation', type=str,
                        help='input train-val annotations path')
    parser.add_argument('-o', '--output', type=str,
                        required=False, default=os.getcwd(),
                        help='Output path to save model, logs and checkpoints')
    parser.add_argument('-ckp', '--ckp_path', type=str,
                        required=False, default=None,
                        help='Input path of a checkpoint to resume training \
                             from it')
    args = parser.parse_args(args=argv)
    return args


def save_model(model, trainer, output, exp_name):
    current_time = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', current_time)

    model_name = '_'.join([timestamp, exp_name, 'model'])
    model_checkpoint = model_name + TORCH_CHECKPOINT_EXT
    model_st_dict = model_name + TORCH_STATE_DICT_EXT
    print('Saving...')
    trainer.save_checkpoint(os.path.join(output, model_checkpoint))
    torch.save(model.state_dict(), os.path.join(output, model_st_dict))
    print('Done')


def dataloader(train_img_path, valid_img_path, annotation_path,
               vocab, batch_size):
    train_data = get_loader(
        root=train_img_path,
        json=annotation_path,
        vocab=vocab,
        transform=transform_function(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count())

    valid_data = get_loader(
        root=valid_img_path,
        json=annotation_path,
        vocab=vocab,
        transform=transform_function(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count())

    return train_data, valid_data


def setup_logs_and_callbacks(output_path, exp_name):
    tensorboard_path = os.path.join(output_path, "logs")
    logs_folder = os.path.join(tensorboard_path, exp_name)
    os.makedirs(logs_folder, exist_ok=True)
    ckpt_path = os.path.join(logs_folder, "-{epoch}")

    logger = TensorBoardLogger(tensorboard_path, exp_name)
    checkpoint_callback = ModelCheckpoint(prefix=exp_name,
                                          filepath=ckpt_path,
                                          monitor='val_loss',
                                          mode="min")
    return logger, checkpoint_callback


def train(args):
    vocab = build_vocab(args.annotation, 4)
    train_loader, valid_loader = dataloader(args.train_img,
                                            args.valid_img,
                                            args.annotation,
                                            vocab,
                                            BATCH_SIZE)
    logger, checkpoint_callback = setup_logs_and_callbacks(args.output,
                                                           args.exp_name)
    trainer = Trainer(resume_from_checkpoint=args.ckpt_path,
                      gpus=1,
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=EARLY_STOP,
                      logger=logger,
                      accumulate_grad_batches=ACC_GRAD_BATCHES,
                      max_epochs=MAX_EPOCHS,
                      val_check_interval=VALIDATION_CHECK,
                      progress_bar_refresh_rate=PROGESS_BAR_RATE,
                      limit_val_batches=LIMIT_VALID_BATCHES,
                      limit_train_batches=LIMIT_TRAIN_BATCHES
                      )

    model = ImageCaptioning(train_loader, valid_loader, vocab)
    print('Start training...')
    trainer.fit(model)
    print('Done')

    return model, trainer


def main(argv):
    args = parse_args(argv)
    args.output = os.path.abspath(args.output)
    model, trainer = train(args)
    save_model(model, trainer)


if __name__ == "__main__":
    main(sys.argv[1:])
