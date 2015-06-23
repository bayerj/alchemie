# -*- coding: utf-8 -*-


import os
import sys
import h5py

import theano
import breze.learn.sgvb as sgvb
from breze.learn import base
from breze.learn.trainer.trainer import Trainer
from breze.learn.trainer.report import OneLinePrinter
import climin.initialize
import numpy as np

from sklearn.grid_search import ParameterSampler

from alchemie.contrib import git_log

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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

    comment = 'some random test comment'

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
        'n_recog_layers': [2,3,4,5,6],
        # number of hidden units per generative or recognition layer, respectively
        'n_gen_hidden_units': [32,64,96,128],
        'n_rec_hidden_units': [32,64,96,128],

        # type of activation/transfer function in generative or recognition part, respectively
        'transfer_gen': ['sigmoid', 'tanh', 'rectifier'],
        'transfer_rec': ['sigmoid', 'tanh', 'rectifier'],

        # latent dimensionality of VAE
        'n_latents': [8,16,32,64],
        # assumption on distribution of latent units
        'latent_assumption': ['DiagGauss'],

        # standard deviation of noise on output (necessary to enforce compression)
        'out_std': [10**-i for i in range(5)],


        # standard deviation of random weight initilization
        'par_init_std': [2**i  for i in range(-6,2)],


        # dropout rate in the input layer or hidden layer(s), respectively
        'p_dropout_inpt': [i/10. for i in range(1,11)],
        'p_dropout_hiddens': [i/10. for i in range(1,11)],

        # maximum training time in minutes
        'minutes': [600],
    }

    sampler = ParameterSampler(grid, n)
    return sampler


def load_data(pars):
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
    return {'train': (X,),
            'val': (VX,),
            'test': (TX,)}


def new_trainer(pars, data):
    #########
    # LOGGING GIT COMMITS
    #########
    modules =  ['theano', 'breze', 'climin', 'alchemie']
    cwd = os.getcwd()
    gl = git_log(modules)
    with open(os.path.join(cwd, 'gitlog.txt'),'w') as result:
        result.write(gl)

    #########
    # BUILDING AND INITIALIZING MODEL FROM pars
    #########
    if pars['latent_assumption'] == 'KW':
        class Assumptions(sgvb.ConstantVarVisibleGaussAssumption, sgvb.KWLatentAssumption):
            out_std = theano.shared(np.array(pars['out_std']).astype(theano.config.floatX))
    else:
        class Assumptions(sgvb.ConstantVarVisibleGaussAssumption, sgvb.DiagGaussLatentAssumption):
            out_std = theano.shared(np.array(pars['out_std']).astype(theano.config.floatX))

    m = sgvb.VariationalAutoEncoder(
        n_inpt=int(data['test'][0].shape[1]),
        n_hiddens_recog=[pars['n_rec_hidden_units']] * pars['n_recog_layers'],
        n_latent = pars['n_latents'],
        n_hiddens_gen = [pars['n_gen_hidden_units']] * pars['n_gen_layers'],
        recog_transfers = [pars['transfer_rec']] * pars['n_recog_layers'],
        gen_transfers = [pars['transfer_gen']] * pars['n_gen_layers'],
        assumptions=Assumptions(),
        batch_size=pars['batch_size'],
        optimizer=pars['optimizer'],
        p_dropout_inpt=pars['p_dropout_inpt'],
        p_dropout_hiddens=pars['p_dropout_hiddens'])

    climin.initialize.randomize_normal(m.parameters.data, 0, pars['par_init_std'])

    #########
    # BUILDING AND INITIALIZING TRAINER
    #########
    n_report = 100
    t = Trainer(
        model=m,
        data=data,
        stop=climin.stops.Any([
                                climin.stops.TimeElapsed(pars['minutes'] * 60),
                                ]),
        pause=climin.stops.ModuloNIterations(n_report),
        report=OneLinePrinter(['n_iter', 'loss', 'val_loss']),
        interrupt=climin.stops.OnSignal())

    return t


def make_report(pars, trainer, data):
    last_pars = trainer.switch_to_best_pars()

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.plot([i['loss'] for i in trainer.infos[0:]])
    fig.savefig('loss.pdf', transparent=True, frameon=False, bbox_inches='tight', pad_inches=.05)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.plot([i['val_loss'] for i in trainer.infos[0:]])
    fig.savefig('val_loss.pdf', transparent=True, frameon=False, bbox_inches='tight', pad_inches=.05)

    ###################
    # RECONSTRUCTION
    ###################
    f_reconstruct = trainer.model.function(['inpt'], 'output')
    X = data['train'][0]
    Y = f_reconstruct(X)

    fig, ax = plt.subplots(3, 1, figsize=(15, 25))

    ax[0].plot(X[:, 0], X[:, 1], 'bx')
    ax[0].plot(Y[:, 0], Y[:, 1], 'rx')
    ax[1].plot(X[:, 0], X[:, 2], 'bx')
    ax[1].plot(Y[:, 0], Y[:, 2], 'rx')
    ax[2].plot(X[:, 1], X[:, 2], 'bx')
    ax[2].plot(Y[:, 1], Y[:, 2], 'rx')
    fig.savefig('reconstruction.pdf', transparent=True, frameon=False, bbox_inches='tight', pad_inches=.05)

    fig, ax3d = plt.subplots(1,1, figsize=(15,11))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot(X[:,0], X[:, 1], X[:, 2], 'bx')
    fig.savefig('reconstruction3d.pdf', transparent=True, frameon=False, bbox_inches='tight', pad_inches=.05)

    ###################
    # SAMPLING
    ###################
    f_generate = trainer.model.function(['sample'], 'output')

    if pars['latent_assumption'] == 'KW':
        qs = np.random.uniform(size=X.shape[0]*pars['n_latents']).reshape((X.shape[0], pars['n_latents'])).astype('float32')
    else: # includes case 'DiagGauss'
        qs = np.random.standard_normal((X.shape[0], pars['n_latents'])).astype('float32')

    S = f_generate(qs)

    alpha=0.1
    fig, ax = plt.subplots(4, 1, figsize=(15, 36))
    ax[0].plot(S[:, 0], S[:, 1], 'ro', alpha=alpha)
    ax[1].plot(S[:, 0], S[:, 2], 'ro', alpha=alpha)
    ax[2].plot(S[:, 1], S[:, 2], 'ro', alpha=alpha)
    ax3d = fig.add_subplot(4, 1, 4, projection='3d')
    ax3d.plot(S[:, 0], S[:, 1], S[:,2], 'ro', alpha=alpha)
    fig.savefig('sampling.pdf', transparent=True, frameon=False, bbox_inches='tight', pad_inches=.05)

    result = {'train_loss': trainer.score(*data['train']),
            'val_loss': trainer.score(*data['val']),
            'test_loss': trainer.score(*data['test'])}

    trainer.switch_to_pars(last_pars)

    return result