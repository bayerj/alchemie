# -*- coding: utf-8 -*-


import cPickle
import glob
import gzip
import os.path
import re


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
    if cp is None:
        fn = 'checkpoint-0.pkl.gz'
    else:
        next_cp_idx = idx_from_checkpoint(cp) + 1
        fn = 'checkpoint-%i.pkl.gz' % next_cp_idx

    with gzip.open(os.path.join(dirname, fn), 'w') as fp:
        cPickle.dump(trainer, fp, protocol=2)
