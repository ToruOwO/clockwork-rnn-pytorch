import numpy as np

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True,
                 rnn_type='rnn'):
        super(RNN, self).__init__()
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=batch_first)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=batch_first)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=batch_first)
        else:
            raise ValueError('Unsupported recurrent network type.')
        self.fc = nn.Linear(hidden_size, input_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

    def forward(self, x):
        # h0 and c0 default to zero
        x, _ = self.rnn(x)

        if self.batch_first:
            B, T, _ = x.shape
            x = x.contiguous().view(B * T, self.hidden_size)
            return self.fc(x).view(B, T, self.input_size)
        else:
            T, B, _ = x.shape
            x = x.permute(1, 0, 2).view(B * T, self.hidden_size)
            return self.fc(x).view(B, T, self.input_size).permute(1, 0, 2)


class CwRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_modules):
        super(CwRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)

        assert hidden_size % n_modules == 0
        self.n_modules = n_modules
        self.module_size = hidden_size // n_modules
        self.module_period = [2 ** t for t in range(n_modules)]

    def step(self, x, hidden, t):
        """Only update block-rows that correspond to the executed modules."""
        hidden_out = torch.zeros_like(hidden).cuda()

        for i in range(self.n_modules):
            start_row_idx = i * self.module_size
            end_row_idx = (i + 1) * self.module_size

            # check if execute current module
            if t % self.module_period[i] == 0:
                xi = torch.mm(x,
                    self.rnn_cell.weight_ih[
                        start_row_idx:end_row_idx].transpose(0, 1))
                xi = torch.add(xi,
                    self.rnn_cell.bias_ih[start_row_idx:end_row_idx])

                # upper triangular matrix mask
                xh = torch.mm(hidden[:, start_row_idx:],
                    self.rnn_cell.weight_hh[
                    start_row_idx:end_row_idx, start_row_idx:].transpose(0, 1))

                xh = torch.add(xh,
                    self.rnn_cell.bias_hh[start_row_idx:end_row_idx])
                xx = torch.tanh(torch.add(xi, xh))

                hidden_out[:, start_row_idx:end_row_idx] += xx

            else:
                hidden_out[:, start_row_idx:end_row_idx] += \
                    hidden[:, start_row_idx:end_row_idx]

        return hidden_out

    def forward(self, x):
        b, t, _ = x.shape
        hidden = torch.zeros(b, self.hidden_size).cuda()  # default to zeros
        x_out = []
        for i in range(t):
            hx = self.step(x[:, i], hidden, i)  # (batch_size, hidden_size)
            hidden = hx
            hx = self.fc(hx)  # (batch_size, output_size)
            x_out.append(hx)

        # output shape (batch_size, seq_len, input_size)
        return torch.stack(x_out, dim=0).permute(1, 0, 2)
