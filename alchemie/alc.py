import cPickle
import gzip
import json
import os
import imp
import importlib
import pprint

import numpy as np

from alchemie import contrib
from breze.learn.utils import JsonForgivingEncoder


def load_module(m):
    """Import and return module identified by ``m``.
    Can be either a string identifying a module or a location of filename of a
    python module."""
    try:
        mod = imp.load_source('mod', m)
    except IOError:
        mod = importlib.import_module(m)
    return mod


def create(args, setup):
    pars = setup.draw_pars(int(args['--amount']))
    # HPC cluster manager starts enumerating tasks with 1.
    for i, p in enumerate(pars, start=1):
        dirname = os.path.join(args['<location>'], str(i))
        os.makedirs(dirname)
        with open(os.path.join(dirname, 'cfg.py'), 'w') as fp:
            fp.write(setup.preamble(i))
            fp.write('\n\n')

            dct_string = pprint.pformat(p)
            fp.write('pars = {\n ')
            fp.write(dct_string[1:-1])
            fp.write('\n}')


def make_trainer(pars, setup, data):
    cps = contrib.find_checkpoints('.')
    if cps:
        print '>>> reloading trainer from checkpoint'
        with gzip.open(cps[-1], 'rb') as fp:
            trainer = cPickle.load(fp)
            trainer.data = data
    else:
        print '>>> creating new trainer'
        trainer = setup.new_trainer(pars, data)

    return trainer


def run(args, setup):
    loc = args['<location>']
    print '>>> changing location to %s' % loc
    os.chdir(loc)

    print '>>> loading data'
    pars = load_module(os.path.join('cfg.py')).pars
    print pars

    data = setup.load_data(pars)
    trainer = make_trainer(pars, setup, data)

    trainer.fit()

    print '>>> making report'
    last_pars = trainer.switch_pars(trainer.best_pars)

    report = setup.make_report(pars, trainer, data)
    trainer.switch_pars(last_pars)

    print '>>> saving to checkpoint'
    idx = contrib.to_checkpoint('.', trainer)

    fn = 'report-last.json' if trainer.stopped else 'report-%i.json' % idx
    with open(fn, 'w') as fp:
        json.dump(report, fp, cls=JsonForgivingEncoder)

    return 0 if trainer.stopped else 9


def evaluate(args, setup):
    directory = os.path.abspath(args['<location>'])
    sub_dirs = [os.path.join(directory, sub_dir)
                for sub_dir in os.listdir(directory)]
    best_loss = np.inf
    best_exp = ''

    for sub_dir in sub_dirs:
        if not os.path.isdir(sub_dir):
            continue
        print '>>> checking %s' % sub_dir
        os.chdir(sub_dir)
        cps = contrib.find_checkpoints(sub_dir)
        if cps:
            with gzip.open(cps[-1], 'rb') as fp:
                trainer = cPickle.load(fp)
                print trainer.best_loss
                if trainer.best_loss < best_loss:
                    best_loss = trainer.best_loss
                    best_exp = sub_dir
        else:
            print '>>> no checkpoints found in this folder.'

    r_string = '>>> found the best experiment in\n>' \
               '>> %s\n>>> with a validation loss of %f' % (best_exp, best_loss)
    print r_string
    with open(os.path.join(directory, 'result.txt'), 'w') as result:
        result.write(r_string)


def main(args, setup):
    if args['create']:
        create(args, setup)
        exit_code = 0
    elif args['evaluate']:
        evaluate(args, setup)
        exit_code = 0
    elif args['run']:
        exit_code = run(args, setup)

    return exit_code
