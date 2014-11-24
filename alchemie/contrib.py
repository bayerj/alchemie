# -*- coding: utf-8 -*-


import cPickle
import glob
import gzip
import imp
import importlib
import os.path
import pprint
import re
import whetlab

from cPickle import PickleError
from os import remove

checkpoint_file_re = re.compile("checkpoint-(\d+).pkl.gz")


def idx_from_checkpoint(fn):
    """Given a checkpoint filename, return the index of it.

    Parameters
    ----------

    fn : string
        Of the form ``checkpoint-<somenumberhere>.pkl.gz``.

    Returns
    -------

    idx : int
        Integer of the index file.

    Examples
    --------

    >>> idx_from_checkpoint('checkpoint-12.pkl.gz')
    12
    >>> idx_from_checkpoint('checkpoint-012.pkl.gz')
    12
    >>> idx_from_checkpoint('checkpoint-1.pkl.gz')
    1
    >>> idx_from_checkpoint('checkpoasdhasint-1.pkl.gz')
    Traceback (most recent call last):
        ...
    ValueError: not a valid checkpoint file naming scheme
    """
    r = checkpoint_file_re.search(fn)
    if r is None:
        raise ValueError('not a valid checkpoint file naming scheme')
    return int(r.groups()[0])


def find_checkpoints(dirname):
    cp_files = glob.glob(os.path.join(dirname, 'checkpoint-*.pkl.gz'))
    cp_files.sort(key=idx_from_checkpoint)
    return cp_files


def latest_checkpoint(dirname):
    cps = find_checkpoints(dirname)
    if cps:
        return cps[-1]
    return None


def to_checkpoint(dirname, trainer):
    cp = latest_checkpoint(dirname)
    rm = False

    if cp is None:
        next_cp_idx = 0
        fn = 'checkpoint-0.pkl.gz'
    else:
        next_cp_idx = idx_from_checkpoint(cp) + 1
        fn = 'checkpoint-%i.pkl.gz' % next_cp_idx
        rm = True

    with gzip.open(os.path.join(dirname, fn), 'w') as fp:
        del trainer.eval_data
        del trainer.val_key

        try:
            cPickle.dump(trainer, fp, protocol=2)
        except PickleError:
            raise
        if rm:
            remove(os.path.join(dirname, cp))

    return next_cp_idx


def load_module(m):
    """Import and return module identified by ``m``.

    Can be either a string identifying a module or a location of filename of a
    python module."""
    try:
        mod = imp.load_source('mod', m)
    except IOError:
        mod = importlib.import_module(m)

    return mod


def write_pars_to_mod(pars, fn):
    with open(fn, 'w') as fp:
        # TODO somehow re introduce job id
        dct_string = pprint.pformat(pars)
        fp.write('pars = {\n ')
        fp.write(dct_string[1:-1])
        fp.write('\n}')


def create(args, mod):
    for i, p in enumerate(pars):
        dirname = os.path.join(args['<location>'], str(i))
        os.makedirs(dirname)
        fn = os.path.join(dirname, 'cfg.py')


def whetlab_get_candidate(experiment_name):
    scientist = whetlab.Experiment(name=experiment_name)
    return scientist.suggest()


def whetlab_submit(experiment_name, pars, result):
    scientist = whetlab.Experiment(name=experiment_name)
    scientist.update(pars, result)
