# -*- coding: utf-8 -*-

import random
import signal

from breze.learn.mlp import Mlp
from breze.learn.trainer.trainer import Trainer
from breze.learn.trainer.report import KeyPrinter, JsonPrinter
import climin.initialize
import numpy as np
from sklearn.grid_search import ParameterSampler


preamble = ''


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
        climin.stops.BetterThan('train_loss', 1e-4),
    ])

    pause = climin.stops.ModuloNIterations(n_report)
    reporter = KeyPrinter(['n_iter', 'train_loss'])

    t = Trainer(
        m,
        stop=stop, pause=pause, report=reporter,
        interrupt=interrupt)
    t.val_key = 'train'
    t.eval_data['train'] = (X, Z)

    return t


def make_report(pars, trainer, data):
    return {'train_loss': trainer.score(*trainer.eval_data['train'])}
