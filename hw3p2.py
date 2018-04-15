#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 13:26:40 2018

@author: Jim Gianoglio
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.base import Callback
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _use_shared_memory

INPUT_DIM = 40
OUTPUT_DIM = 47

def batchify(array, args):
    batch_len = args.batch_len
    batches = array.shape[0] // batch_len
    array = array[:batches * batch_len]
    return array.reshape((batches, batch_len))


def make_inputs(targets):
    # batches: (n, batch_len)
    return np.pad(targets[:, :-1] + 1, [(0, 0), (1, 0)], mode='constant')


def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))

class WsjDataset(Dataset):
    """
    Dataset yields features, assignments, and labels
    """

    def __init__(self, name, args, test=False):
        super(WsjDataset, self).__init__()
        self.name = name
        data = np.load(os.path.join(args.data_directory, '{}.npy'.format(name)))
        if test:
            labels = [np.zeros((d[1].shape[0],), dtype=np.int32) for d in data]
        else:
            labels = np.load(os.path.join(args.data_directory, '{}_phonemes.npy'.format(name)))

        # preprocessing
        self.features = [torch.from_numpy(x[0].T) for x in data]
        self.phonemes = [torch.from_numpy(y).long() for y in labels]
        self.len = len(self.features)

    def __getitem__(self, item):
        return self.features[item], self.phonemes[item]

    def __len__(self):
        return self.len


def wsj_collate_fn(batch):
    """
    Concatenate features, assignments and labels
    """
    padding = 40  # padding between utterances
    # Count totals
    n = len(batch)
    frame_total = padding * 2 * n
    phoneme_total = 0
    for f, a, p in batch:
        frame_total += f.size(1)
        phoneme_total += p.size(0)
    # Allocate storage
    if _use_shared_memory:
        frame_store = batch[0][0].storage()._new_shared(INPUT_DIM * frame_total)
        collated_frames = batch[0][0].new(frame_store).resize_(1, INPUT_DIM, frame_total).zero_()
        phoneme_store = batch[0][1].storage()._new_shared(phoneme_total)
        collated_phonemes = batch[0][1].new(phoneme_store).resize_(phoneme_total).zero_()
    else:
        collated_frames = batch[0][0].new(1, INPUT_DIM, frame_total).zero_()
        collated_phonemes = batch[0][1].new(phoneme_total).zero_()
    # Collate
    framepos = 0
    phonemepos = 0
    for f, p in batch:
        startframe = framepos + padding
        endframe = framepos + padding + f.size(1)
        collated_frames[0, :, startframe:endframe] = f
        collated_phonemes[phonemepos:phonemepos + p.size(0)] = p
        framepos += 2 * padding + f.size(1)
        phonemepos += p.size(0)

    # Return
    return collated_frames, collated_phonemes


class WsjModel(nn.Module):
    def __init__(self, args):
        super(WsjModel, self).__init__()
        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=args.hidden_dim, hidden_size=args.hidden_dim, batch_first=True),
            nn.LSTM(input_size=args.hidden_dim, hidden_size=args.hidden_dim, batch_first=True),
            nn.LSTM(input_size=args.hidden_dim, hidden_size=args.hidden_dim, batch_first=True)])
        self.projection = nn.Linear(in_features=args.hidden_dim, out_features=47)

    def forward(self, features):
        # Model
        h = features
        for l in self.layers:
            h = l(h)
        raw_logits = torch.squeeze(h, 0)
        # Pooling
        oh = make_one_hot(assignments.data)
        sum_logits = torch.mm(raw_logits, oh)  # (46, phonemes)
        sum_assignment = torch.sum(oh, 0)
        pooled_logits = sum_logits / sum_assignment
        pooled_logits = torch.transpose(pooled_logits, 1, 0)  # (phonemes, 46)
        # Return
        return pooled_logits


class CustomLogger(Callback):
    def end_of_training_iteration(self, **_):
        training_loss = self.trainer.get_state('training_loss', default=0)
        training_error = self.trainer.get_state('training_error', default=0)
        print("Training loss: {} error: {}".format(training_loss.numpy()[0], training_error))


def to_text(preds, charset):
    return ["".join(charset[c] for c in line) for line in preds]


def print_generated(lines):
    for i, line in enumerate(lines):
        print("Generated text {}: {}".format(i, line))


class EpochTimer(Callback):
    """
    Callback that prints the elapsed time per epoch
    """

    def __init__(self):
        super(EpochTimer, self).__init__()
        self.start_time = None

    def begin_of_training_run(self, **_kwargs):
        self.start_time = time.time()

    def begin_of_epoch(self, **_kwargs):
        self.start_time = time.time()

    def end_of_epoch(self, epoch_count, **_kwargs):
        assert self.start_time is not None
        end_time = time.time()
        elapsed = end_time - self.start_time
        print("Epoch {} elapsed: {}".format(epoch_count, elapsed))
        self.start_time = None


def train_model(args):
    """
    Performs the training
    """
    if os.path.exists(os.path.join(args.save_directory, Trainer()._checkpoint_filename)):
        # Skip training if checkpoint exists
        return
    model = WsjModel(args)
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
    train_loader = DataLoader(
        WsjDataset('train', args), shuffle=True,
        batch_size=args.batch_size, collate_fn=wsj_collate_fn, **kwargs)
    validate_loader = DataLoader(
        WsjDataset('dev', args), shuffle=True,
        batch_size=args.batch_size, collate_fn=wsj_collate_fn, **kwargs)
    # Build trainer
    trainer = Trainer(model) \
        .build_criterion('CrossEntropyLoss', size_average=False) \
        .build_metric('CategoricalError') \
        .build_optimizer('Adam') \
        .validate_every((1, 'epochs')) \
        .save_every((1, 'epochs')) \
        .save_to_directory(args.save_directory) \
        .set_max_num_epochs(args.epochs) \
        .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                        log_images_every='never'),
                      log_directory=args.save_directory)

    # Bind loaders
    trainer.bind_loader('train', train_loader, num_inputs=2)
    trainer.bind_loader('validate', validate_loader, num_inputs=2)
    trainer.register_callback( EpochTimer())
    if args.cuda:
        trainer.cuda()

    # Go!
    trainer.fit()
    trainer.save()


def main(argv):
    # Training settings
    parser = argparse.ArgumentParser(description='Homework 2 Part 2 Baseline')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size')
    parser.add_argument('--save-directory', type=str, default='output/simple/v1', help='output directory')
    parser.add_argument('--data-directory', type=str, default='', help='data directory')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train')
    parser.add_argument('--hidden_dim', type=int, default=256, metavar='N', help='hidden units')
    parser.add_argument('--num-workers', type=int, default=0, metavar='N', help='number of workers')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    args = parser.parse_args(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    train_model(args)
    write_predictions(args)


if __name__ == '__main__':
    main(sys.argv[1:])