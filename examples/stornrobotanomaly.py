# -*- coding: utf-8 -*-

"""Usage:
    stornrobotanomaly.py create <location> [--amount=<n>]
    stornrobotanomaly.py run <location>
    stornrobotanomaly.py evaluate <location>

Options:
    --amount=<n>        Amount of configurations to create. [default: 1]
    --no-json-log       Do not use JSON for logging.
"""
from alchemie import alc
import docopt


import os
import sys
import h5py
import signal
import time

from theano.printing import pydotprint, pp, debugprint

from breze.learn.base import cast_array_to_local_type
from breze.learn.sgvb import stornrobotanomaly_storns as storns
from breze.learn.data import interleave, padzeros, split
from alchemie.trainer.trainer import Trainer
from alchemie.trainer.report import OneLinePrinter

import climin.initialize
from climin.stops import Any
import numpy as np

from sklearn.grid_search import ParameterSampler


from alchemie.contrib import git_log

from theano.configparser import config_files_from_theanorc as theanorc_path
from shutil import copyfile



class StornRobotAnomaly(object):
    def preamble(self, job_index):
        """Return a string preamble for the the resulting cfg.py file"""
        return ''


    def draw_pars(self, n=1):
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
        class RandomSeed(object):
            def rvs(self):
                return int(np.random.uniform(low=-2**31, high=2**31-1))
        grid = {
            # stochastic gradient algorithm
            'optimizer': ['adadelta'],
            # batch size for SG determination
            'batch_size': [5,10,20],

            # number of generative and recognition layers, respectively
            'n_gen_layers': [1,2,3,4],
            'n_recog_layers': [1,2,3,4],
            # number of hidden units per generative or recognition layer, respectively
            'n_gen_hidden_units': [2**i for i in range(1,6)],
            'n_rec_hidden_units': [2**i for i in range(1,6)],

            # type of activation/transfer function in generative or recognition part, respectively
            'transfer_gen': ['sigmoid', 'tanh', 'rectifier', 'lstm'],
            'transfer_rec': ['sigmoid', 'tanh', 'rectifier', 'lstm'],

            # latent dimensionality of VAE
            'n_latents': [2**i for i in range(1,6)],
            # assumption on distribution of latent units
            'latent_assumption': ['DiagGauss'],

            # standard deviation of noise on output (necessary to enforce compression)
            'out_std': [10**-i for i in range(5)],

            # standard deviation of random weight initilization
            'par_init_std': [2**i  for i in range(-6,2)],
            'par_init_std_in': [0.000001],

            # dropout rate in the input layer or hidden layer(s), respectively
            'p_dropout_inpt': [i/10. for i in range(1,11)],
            'p_dropout_hiddens': [i/10. for i in range(1,11)],
            'p_dropout_hidden_to_out': [i/10. for i in range(1,11)],
            'p_dropout_shortcut': [i/10. for i in range(1,11)],

            # maximum training time in minutes
            'minutes': [300],

            'seed': RandomSeed()

        }

        sampler = ParameterSampler(grid, n)
        return sampler


    def load_data(self, pars):

        datafile = '//brml.tum.de/dfs/public/Datasets/BaxterCollision/data/bluebear/bluebear.h5'
        with h5py.File(datafile) as fp:
            dev_seqs = [np.array(fp['dev'][i]) for  i in fp['dev']]
            txs = [np.array(fp['test'][i]) for  i in fp['test']]
            tzs = np.array(fp['test_anomaly_label'])

        x = dev_seqs[:80]
        m = np.concatenate(x).mean(axis=0)
        s = np.concatenate(x).std(axis=0)

        x = [(xx-m)/s for xx in x]
        vx = [(vxx-m)/s for vxx in dev_seqs[80:]]
        txs = [(tx-m)/s for tx in txs]

        X, M = tuple(interleave(l) for l in padzeros(x, front=False, return_mask=True))
        VX, VM = tuple(interleave(l) for l in padzeros(vx, front=False, return_mask=True))
        TX, TM = tuple(interleave(l) for l in padzeros(txs, front=False, return_mask=True))
        M, VM, TM = M[:,:,:1], VM[:,:,:1], TM[:,:,:1]

        X, M, VX, VM, TX, TM = [cast_array_to_local_type(i) for i in (X, M, VX, VM, TX, TM)]

        return {
            'train': (X,M),
            'val': (VX,VM),
            'test': (TX,TM)
        }


    def new_trainer(self, pars, data):
        #########
        # LOGGING GIT COMMITS
        #########
        modules =  ['theano', 'breze', 'climin', 'alchemie']
        cwd = os.getcwd()
        gl = git_log(modules)
        with open(os.path.join(cwd, 'gitlog.txt'),'w') as result:
            result.write(gl)


        # This should maybe be moved to contrib as a static method;
        # note the different file name format - this is just for documentation purposes
        copyfile(theanorc_path(),os.path.join(cwd,'theanorc.txt'))

        #########
        # BUILDING AND INITIALIZING MODEL FROM pars
        #########
        # if pars['optimizer'] == 'adadelta':
        #     optimizer = 'adadelta', {'step_rate': .1,
        #                     'momentum': 0.99,
        #                     'decay': 0.78,
        #                     'offset': 0.00005}
        # else:
        optimizer = pars['optimizer']

        mystorn = storns.GaussConstVarGaussStorn

        m = mystorn(
            n_inpt=int(data['test'][0].shape[2]),
            n_hiddens_recog=[pars['n_rec_hidden_units']] * pars['n_recog_layers'],
            n_latent=pars['n_latents'],
            n_hiddens_gen=[pars['n_gen_hidden_units']] * pars['n_gen_layers'],
            recog_transfers=[pars['transfer_rec']] * pars['n_recog_layers'],
            gen_transfers=[pars['transfer_gen']] * pars['n_gen_layers'],
            use_imp_weight=True,
            #assumptions=Assumptions(),
            batch_size=pars['batch_size'],
            optimizer=optimizer,
            p_dropout_inpt=pars['p_dropout_inpt'],
            p_dropout_hiddens=pars['p_dropout_hiddens'],
            p_dropout_hidden_to_out=pars['p_dropout_hidden_to_out'],
            p_dropout_shortcut=pars['p_dropout_shortcut']
        )

        #m.shared_std = True
        #m.fixed_std = theano.shared(np.ones((1,)) * .1, broadcastable=(True,))

        m.initialize(
            par_std=pars['par_init_std'],
            par_std_in=pars['par_init_std_in'],
            #sparsify_affine=sparsify_in,
            #sparsify_rec=sparsify_rec,
            spectral_radius=None)

        #m.parameters[m.vae.gen.std][...] = 0.01
        # print 'here'
        # print 'here'
        # print 'here'
        # print type(m.loss)
        # print 'there'
        # with open('Z:/Desktop/test.txt', "w") as file:
        #     #text_file.write("Purchase Amount: %s" % TotalAmount)
        #     debugprint(m.loss, file=file)

        #########
        # BUILDING AND INITIALIZING TRAINER
        #########
        n_report = 10
        t = Trainer(
            model=m,
            data=data,
            stop=climin.stops.Any([
                                    climin.stops.TimeElapsed(pars['minutes'] * 60),
                                    climin.stops.IsNaN(keys=['train_loss', 'val_loss'])
                                    ]),
            pause=climin.stops.ModuloNIterations(n_report),
            report=OneLinePrinter(['n_iter', 'time', 'train_loss', 'val_loss'],spaces=[6,'10.2f','15.8f','15.8f']),
            interrupt=Any([climin.stops.OnSignal()]),#, climin.stops.OnSignal(sig=signal.SIGBREAK)]),
        )#loss_keys=['train', 'val'])

        return t


    def make_report(self, pars, trainer, data):
        last_pars = trainer.switch_pars(trainer.best_pars)

        # fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        # ax.plot([i['loss'] for i in trainer.infos[0:]])
        # fig.savefig('loss.pdf', transparent=True, frameon=False, bbox_inches='tight', pad_inches=.05)
        #
        # fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        # ax.plot([i['val_loss'] for i in trainer.infos[0:]])
        # fig.savefig('val_loss.pdf', transparent=True, frameon=False, bbox_inches='tight', pad_inches=.05)
        #
        # ###################
        # # RECONSTRUCTION
        # ###################
        # f_reconstruct = trainer.model.function(['inpt'], 'output')
        # X = data['train'][0]
        # Y = f_reconstruct(X)
        #
        # fig, ax = plt.subplots(3, 1, figsize=(15, 25))
        #
        # ax[0].plot(X[:, 0], X[:, 1], 'bx')
        # ax[0].plot(Y[:, 0], Y[:, 1], 'rx')
        # ax[1].plot(X[:, 0], X[:, 2], 'bx')
        # ax[1].plot(Y[:, 0], Y[:, 2], 'rx')
        # ax[2].plot(X[:, 1], X[:, 2], 'bx')
        # ax[2].plot(Y[:, 1], Y[:, 2], 'rx')
        # fig.savefig('reconstruction.pdf', transparent=True, frameon=False, bbox_inches='tight', pad_inches=.05)
        #
        # fig, ax3d = plt.subplots(1,1, figsize=(15,11))
        # ax3d = fig.add_subplot(111, projection='3d')
        # ax3d.plot(X[:,0], X[:, 1], X[:, 2], 'bx')
        # fig.savefig('reconstruction3d.pdf', transparent=True, frameon=False, bbox_inches='tight', pad_inches=.05)
        #
        # ###################
        # # SAMPLING
        # ###################
        # f_generate = trainer.model.function(['sample'], 'output')
        #
        # if pars['latent_assumption'] == 'KW':
        #     qs = np.random.uniform(size=X.shape[0]*pars['n_latents']).reshape((X.shape[0], pars['n_latents'])).astype('float32')
        # else: # includes case 'DiagGauss'
        #     qs = np.random.standard_normal((X.shape[0], pars['n_latents'])).astype('float32')
        #
        # S = f_generate(qs)
        #
        # alpha=0.1
        # fig, ax = plt.subplots(4, 1, figsize=(15, 36))
        # ax[0].plot(S[:, 0], S[:, 1], 'ro', alpha=alpha)
        # ax[1].plot(S[:, 0], S[:, 2], 'ro', alpha=alpha)
        # ax[2].plot(S[:, 1], S[:, 2], 'ro', alpha=alpha)
        # ax3d = fig.add_subplot(4, 1, 4, projection='3d')
        # ax3d.plot(S[:, 0], S[:, 1], S[:,2], 'ro', alpha=alpha)
        # fig.savefig('sampling.pdf', transparent=True, frameon=False, bbox_inches='tight', pad_inches=.05)

        result = {'train_loss': trainer.score(*data['train']),
                  'val_loss': trainer.score(*data['val']),
                  'test_loss': trainer.score(*data['test'])}

        trainer.switch_pars(last_pars)
        return result


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    print args
    setup = StornRobotAnomaly()
    sys.exit(alc.main(args, setup))