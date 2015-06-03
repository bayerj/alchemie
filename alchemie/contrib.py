# -*- coding: utf-8 -*-


import cPickle
import glob
import gzip
import os.path
import re

from cPickle import PickleError
from os import remove

from subprocess import Popen, PIPE
import importlib

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
        del trainer.data
        if hasattr(trainer.model, 'assumptions'):
            del trainer.model.assumptions

        try:
            cPickle.dump(trainer, fp, protocol=2)
        except PickleError:
            raise
        if rm:
            remove(os.path.join(dirname,cp))

    return next_cp_idx


def git_log(modules):
    prev_path = os.getcwd()
    gitlog = ''
    for m in modules:
        mod = importlib.import_module(m)
        path = os.path.dirname(mod.__file__)
        os.chdir(path)
        if hasattr(mod,'__version__'):
            info = mod.__version__
        else:
            gitproc = Popen(['git', 'diff-index', 'HEAD'], stdout=PIPE)
            (stdout, _) = gitproc.communicate()
            info = stdout.strip()
            if not info == '':
                info = 'WARNING: unsynced changes in module %s\n' % m + info +'\n\n'

            gitproc = Popen(['git', 'log','-1'], stdout=PIPE)
            (stdout, _) = gitproc.communicate()
            info += stdout.strip()


        gitlog += '%s\n-----\n%s'%(m,info)+'\n\n'

    os.chdir(prev_path)
    return gitlog