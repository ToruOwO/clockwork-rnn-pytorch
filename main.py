"""Train a recurrent network to generate audio signals."""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

from data import AudioSeq
from model import RNN, CwRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--fn', default='data/clair_de_lune.wav', type=str)
parser.add_argument('--n_seq', default=3, type=int)
parser.add_argument('--n_points', default=500, type=int)
parser.add_argument('--start_idx', default=2000000, type=int)
parser.add_argument('--no_input', default=False, type=bool)

parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--n_modules', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.00003)
parser.add_argument('--momentum', type=float, default=0.95)
parser.add_argument('--model', type=str, default='b')
parser.add_argument('--optim', type=str, default='r')
parser.add_argument('--n_epochs', type=int, default=200)

parser.add_argument('--log_per_epoch', type=int, default=1)
parser.add_argument('--save_dir', type=str, default='./outputs')
parser.add_argument('--save_fn', type=str, default='0')
parser.add_argument('--vis', type=bool, default=True)


def plot_losses(losses, save_fn):
    # save losses
    np.save(save_fn + '.npy', losses)

    plt.clf()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.title('Training losses')
    plt.savefig(save_fn + '.png')


def plot_results(y, y_pred, save_fn):
    n_points = y.shape[1]
    for i in range(y.shape[-1]):
        plt.clf()
        for j in range(y.shape[0]):
            plt.plot([k for k in range(j*n_points, (j+1)*n_points)],
                y[j, :, i], '-', alpha=0.5)
            plt.plot([k for k in range(j*n_points, (j+1)*n_points)],
                y_pred[j, :, i], '--', alpha=0.5)

        plt.xlabel('t')
        plt.ylabel('y')

        plt.legend(['GT', 'Pred'])

        plt.savefig(save_fn + f'_{i}.png')


def train(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(os.path.join(args.save_dir, f'{args.save_fn}_args.txt'),
              'w') as fp:
        json.dump(args.__dict__, fp, indent=4)

    if args.model == 'r':
        model = RNN(input_size=2, hidden_size=args.hidden_size).to(device)
    elif args.model == 'b':
        model = RNN(input_size=2, hidden_size=args.hidden_size,
            rnn_type='lstm').to(device)
    elif args.model == 'g':
        model = RNN(input_size=2, hidden_size=args.hidden_size,
            rnn_type='gru').to(device)
    elif args.model == 'c':
        model = CwRNN(input_size=2, hidden_size=args.hidden_size,
            n_modules=args.n_modules).to(device)

    if args.optim == 'a':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'r':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,
            momentum=args.momentum)

    data = AudioSeq(args.n_points, args.start_idx, args.fn, n_seq=args.n_seq)
    print(f'Sampled {args.n_seq} audio sequences at rate {data.rate}, '
          f'each of length {data.signal.shape[1]}.')

    data = torch.Tensor(data.nsignal).to(device)  # (N, T, 2)

    # training
    losses = []
    for epoch in range(args.n_epochs):
        model.train()

        if args.no_input:
            pred = model(torch.zeros_like(data).cuda())
        else:
            pred = model(data)
        loss = F.mse_loss(pred, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation
        valid_loss = 0.0

        model.eval()
        with torch.no_grad():

            if args.no_input:
                pred = model(torch.zeros_like(data).cuda())
            else:
                pred = model(data)
            # pred = model(data)
            loss = F.mse_loss(pred, data)

            valid_loss += loss.item()

        losses.append(valid_loss)

        if (epoch + 1) % args.log_per_epoch == 0:
            print('Epoch [{}/{}], Validation loss: {:.4f}'
                  .format(epoch + 1, args.n_epochs, valid_loss))

    if args.vis:
        # visualize loss
        plot_losses(losses,
            os.path.join(args.save_dir, f'{args.save_fn}_loss'))

        # visualize training results
        data = data.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        plot_results(data, pred,
            os.path.join(args.save_dir, f'{args.save_fn}_plot'))


if __name__ == '__main__':
    arguments = parser.parse_args()
    train(arguments)
