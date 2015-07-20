alchemie
========

Alchemie is a tool to perform machine learning experiments using
[breze](http://github.com/breze-no-salt/breze). It is also designed to work
with cluster environments, such as [slurm](http://slurm.schedmd.com/) using 
Sebastian Urban's [submit](https://github.com/surban/submit).

Alchemie aims to solve the following problems:

 - quick generation of many experiments differing in hyper parameters,
 - interruption and continuation of experiments,
 - generation of rich reports and saving relevant results, e.g., model 
   parameters for future analysis.

Workflow
---------
1. A user will implement a python module representing
an experiment. This module has a certain structure that is explained below.
2. By running
        alc.py create <module> <location> [--amount=<n>]
randomized configurations are generated; each of these configurations corresponds to a directory on the file system contained in ``<location>``, in which a file ``cfg.py`` is placed. 
3. In the next step, the experiment is executed via the command
        alc.py run <module> <location>
The parameters are read from ``cfg.py`` in ``<location>`` (note that the ``<location>`` in the ``run`` command points one level lower than in the ``create`` command, namely to one specific configuration. The experiment will store relevant information in files in the configuration directory.


Writing an experiment module
----------------------------

A alchemie experiment module is expected to implement specific functions. These are ``preamble``, 
``draw_pars``, ``load_data``, ``new_trainer`` and ``make_report``. We will briefly describe their functionality below.

For an example, see the ``examples`` directory of alchemie.

### The preamble
 
Signature:

    def preamble(job_index):
        # ...
        return some_string
        
The idea is to make it possible to add a string prefix to each of the configuration
files ``cfg.py``. It will take an integer as an argument (which is unique in the
set of experiments generated in one call of ``create``) which can serve as meta data.

This is useful for cluster environments, where additional meta data can be
stored in such files.

The ``cfg.py`` file is automatically generated from strings, and should remain executable after adding the preamble, hence the string should be a valid Python comment.



### Drawing parameters
We draw parameters randomly by calling the following function:




    def draw_pars(n=1):
         # ...
         return iterable_of_length_n
         
Here, ``n`` gives the amount of random configurations to draw. ``draw_pars``
will then return an iterable of that length over dictionaries represeting
different parameters. This is compatible with, i.e., ``sklearn.grid_search.ParameterSampler``.

Each configuration is represented by a directory, which has a file ``cfg.py``
in it. This python module contains a dictionary ``pars``,
which fully specifies the configuration needed in the experiment module.

Note that, apart from drawing random parameters, this function can specify (the usage of) any high-level properties of the experiment, such as preprocessing etc. The dictionary generated is passed to every other function involved.

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
                
The function will take the parameters from ``cfg.py`` above. This is particularly of interest if the parameters specify some kind of preprocessing which is done here. To work
with breze trainers, we require that the return value is a dictionary that
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
