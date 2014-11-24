# -*- coding: utf-8 -*-

"""Usage:
    alc.py run <module> <location> [--whetlab-name=<wn>]
    alc.py evaluate <location>

Options:
    --amount=<n>             Amount of configurations to create. [default: 1]
    --no-json-log            Do not use JSON for logging.
    --whetlab-name=<wn>      Name of the whetalb experiment.
"""


import cPickle
import gzip
import json
import pprint
import os
import sys

import docopt
import numpy as np

from alchemie import contrib
from breze.learn.utils import JsonForgivingEncoder
from breze.learn.base import UnsupervisedBrezeWrapperBase


def make_trainer(pars, mod, data):
    cps = contrib.find_checkpoints('.')
    if cps:
        print 'Found checkpoint(s). Load trainer.'
        with gzip.open(cps[-1], 'rb') as fp:
            trainer = cPickle.load(fp)
    else:
        print 'No checkpoints found. Creating new trainer.'
        trainer = mod.new_trainer(pars, data)

    trainer.val_key = 'val'
    trainer.eval_data = data

    return trainer


def run(args, mod):
    # TODO: make sure location exists
    loc = args['<location>']
    new = not os.path.exists(loc)
    if new:
        print 'Creating location %s' % loc
        os.makedirs(loc)

    print 'Changing into directory %s' % loc
    os.chdir(loc)

    # TODO get pars from whetlab or substitute
    if args['--whetlab-name']:
        if new:
            print 'Creating hyper parameters.'
            pars = contrib.whetlab_get_candidate(args['--whetlab-name'])
            contrib.write_pars_to_mod(pars, './parameters.py')
        else:
            pars = contrib.load_module('./parameters.py').pars
    else:
        raise Exception('only whetlab experiments for now')
    print 'Hyper parameters:'
    pprint.pprint(pars)

    print 'Loading data.'
    data = mod.load_data(pars)

    print 'Loading trainer.'
    trainer = make_trainer(pars, mod, data)
    train_data = data['train']

    if isinstance(trainer.model, UnsupervisedBrezeWrapperBase):
        print 'Fitting unsupervised model.'
    else:
        print 'Fitting supervised model.'

    trainer.fit(*train_data)

    print 'Making report.'

    last_pars = trainer.model.parameters.data.copy()
    trainer.model.parameters.data[...] = trainer.best_pars
    report = mod.make_report(pars, trainer, data)
    trainer.model.parameters.data[...] = last_pars

    print 'Saving to checkpoint.'
    idx = contrib.to_checkpoint('.', trainer)

    fn = 'report-last.json' if trainer.stopped else 'report-%i.json' % idx
    with open(fn, 'w') as fp:
        json.dump(report, fp, cls=JsonForgivingEncoder)

    # TODO if stopped, report to whetlab or substitute
    if trainer.stopped:
        print 'Trial has finished.'
        contrib.whetlab_submit(args['--whetlab-name'],
                               pars, trainer.best_loss)
    else:
        print 'Trial is not finished yet.'

    return 0 if trainer.stopped else 9


def evaluate(args):
    dr = os.path.abspath(args['<location>'])
    sub_dirs = [os.path.join(dr, sub_dir) for sub_dir in os.listdir(dr)]
    best_loss = np.inf
    best_exp = ''

    for sub_dir in sub_dirs:
        if not os.path.isdir(sub_dir):
            continue
        os.chdir(sub_dir)
        cps = contrib.find_checkpoints('.')
        if cps:
            print 'Checking %s.' % sub_dir
        with gzip.open(cps[-1], 'rb') as fp:
            trainer = cPickle.load(fp)
            if trainer.best_loss < best_loss:
                best_loss = trainer.best_loss
                best_exp = sub_dir

    r_string = 'Found the best trial in %s with a validation loss of %g' % (best_exp, best_loss)
    print r_string
    with open(os.path.join(dr, 'result.txt'), 'w') as result:
        result.write(r_string)
    return 0


def main(args):
    if args['evaluate']:
        exit_code = evaluate(args)
    elif args['run']:
        mod = contrib.load_module(args['<module>'])
        exit_code = run(args, mod)

    return exit_code


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    print args
    sys.exit(main(args))
