# -*- coding: utf-8 -*-

"""Usage:
    alc.py create <module> <location> [--amount=<n>]
    alc.py run <module> <location>

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

from alchemie import contrib
from breze.learn.utils import JsonForgivingEncoder
from breze.learn.base import SupervisedBrezeWrapperBase
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
            mod.generate_dict(trainer,data)
    else:
        trainer = mod.new_trainer(pars, data)

    return trainer


def run(args, mod):
    loc = args['<location>']
    print '>>> changing location to %s' % loc
    os.chdir(loc)

    print '>>> loading data'
    pars = load_module(os.path.join('./cfg.py')).pars
    data = mod.load_data(pars)
    trainer = make_trainer(pars, mod, data)

    if isinstance(trainer.model, UnsupervisedBrezeWrapperBase):
        print '>>> Fitting unsupervised model'
        trainer.fit(data[0])
    else:
        print '>>> Fitting supervised model'
        trainer.fit(data[0],data[1])

    print '>>> making report'
    report = mod.make_report(pars, trainer, data)

    print '>>> saving to checkpoint'
    idx = contrib.to_checkpoint('.', trainer)

    fn = 'report-last.json' if trainer.stopped else 'report-%i.json' % idx
    with open(fn, 'w') as fp:
        json.dump(report, fp, cls=JsonForgivingEncoder)



    return 0 if trainer.stopped else 9


def evaluate(args, mod):
    dir = args['location']
    sub_dirs = [os.path.join(dir, sub_dir)
                       for sub_dir in os.listdir(dir)]
    for sub_dir in sub_dirs:
        last_report = os.path.join(sub_dir,'report-last.json')
        if os.path.isfile(last_report):
            with open(last_report, 'r') as f:
                f.




def main(args):
    # Get module.
    mod = load_module(args['<module>'])

    if args['create']:
        create(args, mod)
        exit_code = 0
    elif args['run']:
        exit_code = run(args, mod)

    return exit_code


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    print args
    sys.exit(main(args))
