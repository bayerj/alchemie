alchemie
========

Alchemie is a tool to perform machine learning experiments using
[breze](http://github.com/breze-no-salt/breze). It is also designed to work
with cluster environments, such as [slurm](http://slurm.schedmd.com/) using 
Sebastian Urban's [submit](https://github.com/surban/submit).

Alchemie aims to solve the following problems:

 - quick generation of many experiments differing in hyper parameters,
 - interruption and continuation of experiments,
 - generationg of rich reports and saving of relevant results, e.g. model 
   parameters for future analysis.

The work flow is as follows. A user will implement a python module representing
an experiment. She will then generate randomized configurations in a
first step; each of these configurations is then represented as a directory on
the file system, in which a file ``cfg.py`` is placed. In the next step, he will
run the experiment. The experiment will store relevant information in files in
the configuration directory.


Writing an experiment module
----------------------------

A module is expected to have various functions in it. These are ``preamble``, 
``draw_pars``, ``load_data``, ``new_trainer`` and ``make_report``. We will go
through each of them.

For an example, see ``examples/mlpxor.py``.

### The preamble
 
Signature:

    def preamble(job_index):
        # ...
        return some_string
        
The idea is to make it possible to add a prefix to each of the configuration
files generated. It will take an integer as an argument (which is unique in the
set of experiments generated in one run) which can serve as meta data.

This is useful for cluster environments, where additional meta data can be
placed in such files.

### Drawing parameters

Each configuration is represented by a directory, which has a file ``cfg.py``
in it. This is supposed to be a python module containing a dictionary ``pars``,
which fully specifies the configuration in conjunction with the experiment
module. This dictionary will typically be randomly generated; this is supposed
to be done with the following function.

    def draw_pars(n=1):
         # ...
         return iterable_of_length_n
         
Here, ``n`` gives the amount of random configurations to draw. ``draw_pars``
will then return an iterable of that length over dictionaries represeting
different parameters.

### Loading data

Each machine learning experiment works on data. During each startup and
resumption of an experiment, we will need to load this data and make it
available to the training process. The signature of the function is as follows:

    def load_data(pars):
        # ...
        return {'train': some_train_data,     # required
                'val': some_validation_data,  # required
                'test': some_testing_data,    # not required
               }
                
The function will take the parameters from ``draw_pars`` above. This is e.g. if
the parameters indicate some kind of preprocessing which is done here. To work
with breze trainer's, we require that the return value is a dictionary that
works for the ``.eval_data`` field of a breze ``Trainer`` object. The ``val``
field will be used for validation during training.

It makes sense to populate the dictionary also with testing data, so that we
can have access to it later.

### Making a new trainer (and model)

This function is to create a new trainer for each configuration, and the model
under consideration along with it. 

    def new_trainer(pars, data):
        # ...
        return trainer
        
This is the place to do things like:

 - Create a model,
 - Initialize its parameters,
 - Set stopping, pausing and interruption criterions.
 
Note that you can make use of the data dictionary to fully specify things here,
     e.g. making use of a varying input dimensionality.

### Reporting

After each job running, alchemie will run ``make_report`` and save the
results to a json file. If the job was interrupted, it will be numbered
consecutively, e.g. ``report-1.json``. Otherwise it will be called 
``report-last.json``. The signature is as follows:

    def make_report(pars, trainer, data):
        # ...
        return {'test_loss': 0}   # You want more useful info here.
   			
   			
## Further readings

The code is very short--and there is an example in the ``examples`` directory.
