import os.path
import shutil
import sys
import numpy as np

import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['ps.useafm'] = True
mpl.rcParams.update({'font.size': 14})
mpl.rc('axes', labelsize=14)
mpl.rc('ytick', labelsize=14)
mpl.rc('xtick', labelsize=14)

from pylab import *

import experiments
import pickle

################################################################################################
# Read experiment to run
################################################################################################

ID = int(sys.argv[1:][0])

opt = experiments.opt[ID]

crop_size = int(sys.argv[1:][1])

# Skip execution if instructed in experiment
if opt.skip:
    print("SKIP")
    quit()

for num_iter in range(100):
    with open(opt.log_dir_base + opt.name + '/maps/top/' + str(experiments.crop_sizes[crop_size])
              + '/' + str(num_iter) + '.pkl', 'rb') as f:
        results = pickle.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.imshow(results)
    plt.savefig(opt.log_dir_base + opt.name + '/maps/top/' + str(experiments.crop_sizes[crop_size])
              + '/' + str(num_iter) + '.pdf', format='pdf', dpi=1000)

    plt.close('all')

    with open(opt.log_dir_base + opt.name + '/maps/confidence/' + str(experiments.crop_sizes[crop_size])
              + '/' + str(num_iter) + '.pkl', 'rb') as f:
        results = pickle.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.imshow(results)
    plt.savefig(opt.log_dir_base + opt.name + '/maps/confidence/' + str(experiments.crop_sizes[crop_size])
              + '/' + str(num_iter) + '.pdf', format='pdf', dpi=1000)

    plt.close('all')