"""Preprocess wav data to be fed into recurrent networks."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile


class AudioSeq(object):
    ENCODING_VALUES = [2, 4, 8, 16, 32]

    def __init__(self, n_points, start_idx, fn, n_seq=1, encoding=16):
        # TODO : Find a better way to initialize values
        assert encoding in self.ENCODING_VALUES, "".format(
            "encoding values should be in {}".format(self.ENCODING_VALUES))
        assert isinstance(fn, str), 'wrong filename type {}'.format(type(fn))
        self.fn = fn
        self.rate, self.signal = scipy.io.wavfile.read(fn)

        # get signal
        assert (0 <= start_idx < len(self.signal))
        self.signal = self.signal[start_idx:start_idx + n_points * n_seq]
        self.signal = np.reshape(self.signal, (n_seq, n_points, -1))
        # self.signal = np.stack(np.split(self.signal, n_seq))

        # get normalized signal
        self.normalize()

    def normalize(self, encoding=16, verbose=False):
        # Normalization between -1.0 and 1.0
        self.nsignal = self.signal.astype(np.int64)
        M = np.amax(self.nsignal)
        m = np.amin(self.nsignal)
        print(f'Max {M}, min {m} before Normalization.')

        self.nsignal = -1.0 + 2.0 * (self.nsignal - float(m)) / float(M - m)
        if verbose:
            print(self.nsignal)

    def show_signal(self, signal=None, stype='normalized'):
        fs = ['-', '--', '-.', '-x', '-o', '-^']
        if signal == None:
            if stype == 'normalized':
                signal = self.nsignal
            else:
                signal = self.signal
        n_points = signal.shape[1]
        plt.figure()
        for i, s in enumerate(signal):
            plt.plot([j for j in range(i*n_points, (i+1)*n_points)], s, fs[i])
        plt.show()


def print_specs(args):
    p = AudioSeq(args.n_points, args.start_idx, args.fn, args.n_seq)

    print("sample rate")
    print(p.rate)

    print("number of points")
    print(p.signal.shape[1])

    p.normalize()
    print("normalized signal")

    p.show_signal()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--fn', default='audio_data/clair_de_lune.wav',
        help='Path to raw audio data', type=str)
    parser.add_argument('--n_seq', default=3,
        help='Number of signal sequences to sample', type=int)
    parser.add_argument('--n_points', default=500,
        help="Number of points to consider in the signal", type=int)
    parser.add_argument('--start_idx', default=2000000,
        help="starting position of the signal", type=int)
    args = parser.parse_args()
    # print(args)

    print_specs(args)