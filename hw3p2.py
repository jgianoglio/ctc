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
from operator import itemgetter

import numpy as np
import torch
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.base import Callback
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _use_shared_memory
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from warpctc_pytorch import CTCLoss

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
        #self.features = [torch.from_numpy(x.T) for x in data]
        self.features = [torch.from_numpy(x) for x in data]
        self.phonemes = [torch.from_numpy(y).long() for y in labels]
        self.len = len(self.features)

    def __getitem__(self, item):
        return self.features[item], self.phonemes[item]

    def __len__(self):
        return self.len

def getKey(item):
    return item[0].size()[0]

def wsj_collate_fn(batch):
    """
    Concatenate features and labels

    - Sort the batch by the # of frames in each utterance in decending order. Should have 32 utterances (i.e. batch size)
    - Create a tensor of zeros the size of (max Utt. Len X batch size)

    """
    # Count totals
    batch_size = len(batch)  # Should be 32

    # Sort the batch in descending order by length of utterance
    sorted_batch = sorted(batch, key=getKey, reverse=True)

    max_length = sorted_batch[0][0].size()[0]
    phoneme_total = 0
    for f, p in batch:
        #frame_total += f.size(1)
        phoneme_total += p.size(0)
    # Allocate storage
    if _use_shared_memory:
        frame_store = sorted_batch[0][0].storage()._new_shared(max_length * batch_size * 40)
        collated_frames = sorted_batch[0][0].new(frame_store).resize_(max_length, batch_size, 40).zero_()
        phoneme_store = sorted_batch[0][1].storage()._new_shared(phoneme_total)
        collated_phonemes = sorted_batch[0][1].new(phoneme_store).resize_(phoneme_total).zero_()
        #numFrame_store = sorted_batch[0][0].storage()._new_shared(batch_size)
        num_frames = torch.IntTensor(batch_size,).zero_()
        num_phonemes = torch.IntTensor(batch_size).zero_()
    else:
        collated_frames = sorted_batch[0][0].new(max_length, batch_size, 40).zero_()
        collated_phonemes = sorted_batch[0][1].new(phoneme_total).zero_()
        num_frames = torch.IntTensor(batch_size,).zero_()
        num_phonemes = torch.IntTensor(batch_size).zero_()
    # Collate
    framepos = 0
    phonemepos = 0
    for counter, f in enumerate(sorted_batch):
        #startframe = framepos + padding
        #endframe = framepos + padding + f.size(1)
        length = f[0].size(0)
        collated_frames[:length, counter, :] = f[0]
        num_frames[counter] = length
        num_phonemes[counter] = f[1].size(0)

        collated_phonemes[phonemepos:phonemepos + f[1].size(0)] = f[1]
        #framepos += 2 * padding + f.size(1)
        phonemepos += f[1].size(0)

    # Return
    return collated_frames, num_frames, num_phonemes, collated_phonemes


class WsjModel(nn.Module):
    def __init__(self, args):
        super(WsjModel, self).__init__()
        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=INPUT_DIM, hidden_size=args.hidden_dim),
            nn.LSTM(input_size=args.hidden_dim, hidden_size=args.hidden_dim),
            nn.LSTM(input_size=args.hidden_dim, hidden_size=args.hidden_dim)])
        self.projection = nn.Linear(in_features=args.hidden_dim, out_features=47)

    def forward(self, input, num_frames, num_phonemes):
        h = input  # (n, t)
        h = pack_padded_sequence(h, num_frames.data.numpy())
        #h = self.embedding(h)  # (n, t, c)
        states = []
        for rnn in self.rnns:
            h, state = rnn(h)
            states.append(state)
        h, _ = pad_packed_sequence(h)
        h = self.projection(h)
        #if stochastic:
        #    gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
        #    h += gumbel
        logits = h

        return logits, num_frames, num_phonemes

class ctc_loss(CTCLoss):
    def __init__(self):
        super(ctc_loss, self).__init__()

    def forward(self, model_output, labels):
        """
        acts: Tensor of (seqLength x batch x outputDim) containing output from network
        labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        act_lens: Tensor of (batch) containing label length of each example
        """
        labels = labels.cpu().int()
        acts = model_output[0]
        act_lens = model_output[1].cpu()
        label_lens = model_output[2].cpu()
        #print(model_output)
        assert len(labels.size()) == 1  # labels must be 1 dimensional
        #_assert_no_grad(labels)
        #_assert_no_grad(act_lens)
        #_assert_no_grad(label_lens)
        return self.ctc(acts, labels, act_lens, label_lens, self.size_average,
                        self.length_average)


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
        .build_criterion(ctc_loss()) \
        .build_metric(ctc_loss()) \
        .build_optimizer('Adam') \
        .validate_every((1, 'epochs')) \
        .save_every((1, 'epochs')) \
        .save_to_directory(args.save_directory) \
        .set_max_num_epochs(args.epochs) \
        #.build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                        #log_images_every='never'),
                      #log_directory=args.save_directory)

    # Bind loaders
    trainer.bind_loader('train', train_loader, num_inputs=3)
    trainer.bind_loader('validate', validate_loader, num_inputs=3)
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