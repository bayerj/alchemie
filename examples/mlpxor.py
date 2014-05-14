# -*- coding: utf-8 -*-

import random
import signal
import os

from breze.learn.mlp import Mlp
from breze.learn.trainer.trainer import Trainer
from breze.learn.trainer.report import KeyPrinter, JsonPrinter
import climin.initialize
import numpy as np
from sklearn.grid_search import ParameterSampler



def preamble(job_index):
    train_folder = os.path.dirname(os.path.realpath(__file__))
    module = os.path.join(train_folder, 'mlpxor.py')
    script = os.path.join(train_folder, '../scripts/alc.py')
    runner = 'python %s run %s' % (script, module)

    pre = '#SUBMIT: runner=%s\n' % runner
    pre += '#SUBMIT: gpu=no\n'

    minutes_before_3_hour = 15
    slurm_preamble = '#SBATCH -J MLPXOR_%d\n' % (job_index)
    slurm_preamble += '#SBATCH --mem=4000\n'
    slurm_preamble += '#SBATCH --signal=INT@%d\n' % (minutes_before_3_hour*60)
    return pre + slurm_preamble

def draw_pars(n=1):
    class OptimizerDistribution(object):
        def rvs(self):
            grid = {
                'step_rate': [0.0001, 0.0005, 0.005],
                'momentum': [0.99, 0.995],
                'decay': [0.9, 0.95],
            }

            sample = list(ParameterSampler(grid, n_iter=1))[0]
            sample.update({'step_rate_max': 0.05, 'step_rate_min': 1e-5})
            return 'rmsprop', sample

    grid = {
        'n_hidden': [3],
        'hidden_transfer': ['sigmoid', 'tanh', 'rectifier'],

        'par_std': [1.5, 1, 1e-1, 1e-2],

        'optimizer': OptimizerDistribution(),
    }

    sampler = ParameterSampler(grid, n)
    return sampler


def load_data(pars):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Z = np.array([0, 1, 1, 0]).reshape((4, 1))

    return X, Z

def make_data_dict(trainer,data):
    trainer.val_key = 'train'
    trainer.eval_data = {}
    trainer.eval_data['train'] = ([data[0],data[1]])


def new_trainer(pars, data):
    X, Z = data
    m = Mlp(2, [pars['n_hidden']], 1,
            hidden_transfers=[pars['hidden_transfer']], out_transfer='sigmoid',
            loss='bern_ces',
            optimizer=pars['optimizer'])
    climin.initialize.randomize_normal(m.parameters.data, 0, pars['par_std'])

    n_report = 100

    interrupt = climin.stops.OnSignal()
    print dir(climin.stops)
    stop = climin.stops.Any([
        climin.stops.AfterNIterations(10000),
        climin.stops.OnSignal(signal.SIGTERM),
        climin.stops.NotBetterThanAfter(1e-1,500,key='train_loss'),
    ])

    pause = climin.stops.ModuloNIterations(n_report)
    reporter = KeyPrinter(['n_iter', 'train_loss'])

    t = Trainer(
        m,
        stop=stop, pause=pause, report=reporter,
        interrupt=interrupt)

    make_data_dict(t, data)

    return t


def make_report(pars, trainer, data):
    return {'train_loss': trainer.score(*trainer.eval_data['train'])}
