# -*- coding: utf-8 -*-

import signal
import os, sys

from time import sleep

import h5py


from breze.learn.sgvb import VariationalAutoEncoder as Vae
from breze.learn import base
from breze.learn.trainer.trainer import Trainer
from breze.learn.trainer.report import OneLinePrinter
import climin.initialize
import numpy as np
from sklearn.grid_search import ParameterSampler


def preamble(job_index):
    """Return a string preamble for the the resulting cfg.py file"""

    train_folder = os.path.dirname(os.path.realpath(__file__))
    module = os.path.join(train_folder, 'mlpxor.py')
    script = os.path.join(os.path.dirname(train_folder), 'scripts', 'alc.py')
    runner = 'python %s run %s' % (script, module)

    pre = '#SUBMIT: runner=%s\n' % runner
    pre += '#SUBMIT: gpu=no\n'

    minutes_before_3_hour = 15
    slurm_preamble = '#SBATCH -J MLPXOR_%d\n' % (job_index)
    slurm_preamble += '#SBATCH --mem=4000\n'
    slurm_preamble += '#SBATCH --signal=INT@%d\n' % (minutes_before_3_hour*60)

    comment = ''

    return pre + slurm_preamble + ('\n\n"""\n' + comment + '\n"""\n' if not comment=='' else '')


def draw_pars(n=1):
    """
    :param n: number of random configurations
    :return: sklearn ParameterSampler object producing n random samples from the specified grid
    """
    # class OptimizerDistribution(object):
    #     def rvs(self):
    #         grid = {
    #             'step_rate': [0.0001, 0.0005, 0.005],
    #             'momentum': [0.99, 0.995],
    #             'decay': [0.9, 0.95],
    #         }
    #
    #         sample = list(ParameterSampler(grid, n_iter=1))[0]
    #         sample.update({'step_rate_max': 0.05, 'step_rate_min': 1e-5})
    #         return 'rmsprop', sample

    grid = {
        # stochastic gradient algorithm
        'optimizer': ['adam', 'adadelta'],
        # batch size for SG determination
        'batch_size': [10,25,50],

        # number of generative and recognition layers, respectively
        'n_gen_layers': [2,3,4,5,6],
        # 'n_recog_layers': [2,3,4,5,6],
        # number of hidden units per generative or recognition layer, respectively
        'n_gen_hidden_units': [32,64,96,128],
        # 'n_rec_hidden_units'] = [32,64,96,128],

        # type of activation/transfer function in generative or recognition part, respectively
        'transfer_gen': ['sigmoid', 'tanh', 'rectifier'],
        #['transfer_rec'] = ['sigmoid', 'tanh', 'rectifier'],

        # latent dimensionality of VAE
        'n_latents': [8,16,32,64],
        # assumption on distribution of latent units
        'latent_assumption': ['DiagGauss'],

        # standard deviation of random weight initilization
        'init_std': [2**i  for i in range(-6,2)],

        # dropout rate in the input layer or hidden layer(s), respectively
        'p_dropout_inpt': [i/10. for i in range(1,11)],
        'p_dropout_hiddens': [i/10. for i in range(1,11)],

        # maximum training time in minutes
        'minutes': [600],
    }

    sampler = ParameterSampler(grid, n)
    return sampler


def load_data(pars):
    sys.stdout.write('Loading data... ')
    sys.stdout.flush()

    with h5py.File('P:/Datasets/BaxterCollision/data/bluebear/bluebear.h5') as fp:
    #with h5py.File('/Users/bayerj/brmlpublic/Datasets/BaxterCollision/data/bluebear/bluebear.h5') as fp:
        dev_seqs = [np.array(fp['dev'][i]) for  i in fp['dev']]
        txs = [np.array(fp['test'][i]) for  i in fp['test']]
        tzs = np.array(fp['test_anomaly_label'])

    X = np.concatenate(dev_seqs[:75])
    VX = np.concatenate(dev_seqs[75:])
    TX = np.concatenate(txs)


    # restriction to three dimensions
    X = X[:, (0, 1, 2)]
    VX = VX[:, (0, 1, 2)]
    TX = TX[:, (0, 1, 2)]


    m, s = np.mean(X,axis=0), np.std(X, axis=0)

    X = (X - m) / s
    VX = (VX - m) / s
    TX = (TX - m) / s

    X, VX, TX = [base.cast_array_to_local_type(i) for i in (X, VX, TX)]
    sys.stdout.write('Done.\n')
    sys.stdout.flush()

    return {'train': X,
            'val': VX,
            'test': TX}


def new_trainer(pars, data):
    m = Vae(int(data['test'].shape[1],
            [pars['n_hidden']], 1,
            hidden_transfers=[pars['hidden_transfer']], out_transfer='sigmoid',
            loss='bern_ces',
            optimizer=pars['optimizer'])
    climin.initialize.randomize_normal(m.parameters.data, 0, pars['par_std'])

    n_report = 100

    interrupt = climin.stops.OnSignal()
    stop = climin.stops.Any([
        climin.stops.AfterNIterations(10000),
        climin.stops.OnSignal(signal.SIGTERM),
        climin.stops.NotBetterThanAfter(1e-1, 5000, key='train_loss'),
    ])

    pause = climin.stops.ModuloNIterations(n_report)
    reporter = OneLinePrinter(['n_iter', 'train_loss', 'val_loss'])

    t = Trainer(
        m,
        stop=stop, pause=pause, report=reporter,
        interrupt=interrupt)

    t.val_key = 'val'
    t.eval_data = data

    return t


def make_report(pars, trainer, data):
    return {'train_loss': trainer.score(*trainer.eval_data['train']),
            'val_loss': trainer.score(*trainer.eval_data['val']),
            'test_loss': trainer.score(*trainer.eval_data['test'])}
