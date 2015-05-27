# -*- coding: utf-8 -*-

"""Usage:
    alc.py create <module> <location> [--amount=<n>]
    alc.py run <module> <location>
    alc.py evaluate <location>

Options:
    --amount=<n>        Amount of configurations to create. [default: 1]
    --no-json-log       Do not use JSON for logging.
"""


import cPickle
import gzip
import imp
import importlib
import json
import os
import pprint
import sys

import docopt
import numpy as np

from alchemie import contrib
from breze.learn.utils import JsonForgivingEncoder
from breze.learn.base import UnsupervisedBrezeWrapperBase


def load_module(m):
    """Import and return module identified by ``m``.

    Can be either a string identifying a module or a location of filename of a
    python module."""
    try:
        mod = imp.load_source('mod', m)
    except IOError:
        mod = importlib.import_module(m)

    return mod


def create(args, mod):
    pars = mod.draw_pars(int(args['--amount']))
    for i, p in enumerate(pars):
        dirname = os.path.join(args['<location>'], str(i))
        os.makedirs(dirname)
        with open(os.path.join(dirname, 'cfg.py'), 'w') as fp:
            fp.write(mod.preamble(i))
            fp.write('\n\n')

            dct_string = pprint.pformat(p)
            fp.write('pars = {\n ')
            fp.write(dct_string[1:-1])
            fp.write('\n}')


def make_trainer(pars, mod, data):
    cps = contrib.find_checkpoints('.')
    if cps:
        with gzip.open(cps[-1], 'rb') as fp:
            trainer = cPickle.load(fp)
    else:
        trainer = mod.new_trainer(pars, data)

    trainer.val_key = 'val'
    trainer.eval_data = data

    return trainer


def run(args, mod):
    loc = args['<location>']
    print '>>> changing location to %s' % loc
    os.chdir(loc)

    print '>>> loading data'
    pars = load_module(os.path.join('cfg.py')).pars
    print pars
    data = mod.load_data(pars)
    trainer = make_trainer(pars, mod, data)
    train_data = data['train']

    if isinstance(trainer.model, UnsupervisedBrezeWrapperBase):
        print '>>> Fitting unsupervised model'
    else:
        print '>>> Fitting supervised model'

    trainer.fit(*train_data)

    print '>>> making report'

    last_pars = trainer.model.parameters.data.copy()
    trainer.model.parameters.data[...] = trainer.best_pars
    report = mod.make_report(pars, trainer, data)
    trainer.model.parameters.data[...] = last_pars

    print '>>> saving to checkpoint'
    idx = contrib.to_checkpoint('.', trainer)

    fn = 'report-last.json' if trainer.stopped else 'report-%i.json' % idx
    with open(fn, 'w') as fp:
        json.dump(report, fp, cls=JsonForgivingEncoder)

    return 0 if trainer.stopped else 9


def evaluate(args):
    dir = os.path.abspath(args['<location>'])
    sub_dirs = [os.path.join(dir, sub_dir)
                       for sub_dir in os.listdir(dir)]
    best_loss = np.inf
    best_exp = ''

    for sub_dir in sub_dirs:
        if not os.path.isdir(sub_dir):
            continue
        os.chdir(sub_dir)
        cps = contrib.find_checkpoints('.')
        if cps:
            print '>>> checking %s' %sub_dir
        with gzip.open(cps[-1], 'rb') as fp:
                trainer = cPickle.load(fp)
                if trainer.best_loss < best_loss:
                    best_loss = trainer.best_loss
                    best_exp = sub_dir

    r_string = '>>> found the best experiment in\n>>> %s\n>>> with a validation loss of %f' %(best_exp, best_loss)
    print r_string
    with open(os.path.join(dir, 'result.txt'),'w') as result:
        result.write(r_string)


def main(args):

    if args['create']:
        mod = load_module(args['<module>'])
        create(args, mod)
        exit_code = 0
    elif args['evaluate']:
        evaluate(args)
        exit_code = 0
    elif args['run']:
        mod = load_module(args['<module>'])
        exit_code = run(args, mod)

    return exit_code


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    print args
    sys.exit(main(args))
