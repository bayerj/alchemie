# -*- coding: utf-8 -*-

"""Usage:
    mlpxor.py create <location> [--amount=<n>]
    mlpxor.py run <location>
    mlpxor.py evaluate <location>

Options:
    --amount=<n>        Amount of configurations to create. [default: 1]
"""

import sys

import docopt
import numpy as np
from sklearn.grid_search import ParameterSampler

from alchemie import alc
from alchemie.contrib import git_log, copy_theanorc

from breze.learn.mlp import Mlp
from breze.learn.trainer.trainer import Trainer
from breze.learn.trainer.report import OneLinePrinter

import climin.initialize
from climin.stops import Any


class MlpXOR(object):
    def preamble(self, job_index):
        """Return a string preamble for the the resulting cfg.py file"""
        preamble = '"""Here goes any comment you want to add to the cfg\n\n"""'
        return preamble

    def draw_pars(self, n=1):
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

    def load_data(self, pars):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Z = np.array([0, 1, 1, 0]).reshape((4, 1))

        return {'train': (X, Z),
                'val': (X, Z),
                'test': (X, Z)}

    def new_trainer(self, pars, data):
        modules = ['theano', 'breze', 'climin', 'alchemie']
        git_log(modules)
        copy_theanorc()

        m = Mlp(2, [pars['n_hidden']], 1,
                hidden_transfers=[pars['hidden_transfer']],
                out_transfer='sigmoid',
                loss='bern_ces',
                optimizer=pars['optimizer'])
        climin.initialize.randomize_normal(
            m.parameters.data, 0, pars['par_std'])

        n_report = 100

        t = Trainer(
            model=m,
            data=data,
            stop=climin.stops.Any([
                climin.stops.AfterNIterations(10000),
                climin.stops.NotBetterThanAfter(1e-1, 5000, key='val_loss')]
            ),
            pause=climin.stops.ModuloNIterations(n_report),
            report=OneLinePrinter(
                keys=['n_iter', 'runtime', 'train_loss', 'val_loss'],
                spaces=[6, '10.2f', '15.8f', '15.8f']
            ),
            interrupt=climin.stops.OnSignal(),
        )

        return t

    def make_report(self, pars, trainer, data):
        last_pars = trainer.switch_pars(trainer.best_pars)

        result = {'train_loss': trainer.score(*data['train']),
                  'val_loss': trainer.score(*data['val']),
                  'test_loss': trainer.score(*data['test'])}

        trainer.switch_pars(last_pars)
        return result

if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    print args
    setup = MlpXOR()
    sys.exit(alc.main(args, setup))
