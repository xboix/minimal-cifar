import numpy as np
#import settings
import sys

import os.path

import pandas as pd
import seaborn


import matplotlib as mpl
import seaborn

import experiments

seaborn.set()
seaborn.set_style("whitegrid")
seaborn.set_context("poster")

import matplotlib as mpl
from matplotlib import pyplot
from matplotlib import rc
#rc('text', usetex=True)

import itertools
import seaborn as sns

PATH_TO_DATA = "./results/" #"/om/user/xboix/share/minimal-images/"
#"



TOTAL = 10000

all_nets = [experiments.opt[i+1].name for i in range(5)]
crops = [28, 24, 20, 16, 12]

name_nets = ['28 pix.', '$\geq$24 pix.',
             '$\geq$20 pix.', '$\geq$16 pix.',
             '$\geq$12 pix.']

mm = np.zeros([len(all_nets), len(crops)])
for idx_net, nets in enumerate(all_nets):
    tmp = np.load(PATH_TO_DATA + 'tmp_results_accuracy' + nets + '.npy')
    for idx_metric, crop_metric in enumerate([28, 24, 20, 16, 12]):
        mm[idx_net][idx_metric] = tmp[idx_metric]

cc = itertools.cycle(sns.cubehelix_palette(8))
fig, ax = pyplot.subplots()
for idx_net, nets in enumerate(['3', '8', '13', '18', '23']):
    q = np.zeros(5)
    s = np.zeros(5)
    for i, _ in enumerate(crops):
        q[i] = mm[idx_net][i]
    if idx_net==0:
        pyplot.errorbar(x=crops, y=100*q, yerr=10*s, fmt='-o',label=name_nets[idx_net], linewidth=4)
    else:
        pyplot.errorbar(x=crops, y=100 * q, yerr=10 * s, fmt='-o', label=name_nets[idx_net], linewidth=4, color=next(cc))

    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

pyplot.xticks([28, 24, 20, 16, 12], ['28', '24', '20', '16', '12'])


ax.legend(loc='lower right', frameon= True, title="Crops at Training")
ax.set_xlabel('Crop Size')
ax.set_ylabel('Accuracy (%)')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(25)

pyplot.gcf().subplots_adjust(bottom=0.17, top=0.96, left=0.16, right=0.96)


pyplot.savefig('./plots/accuracy/accuracy_augmentation.pdf', dpi=1000)




################################################################################################################

all_nets = [experiments.opt[i+6].name for i in range(5)]

name_nets = ['Non Regularized', 'Data augment.',
             'Dropout', 'Weight Decay',
             'All Regularizers']

colors = ["amber", "greyish", "orange", "black"]

mm = np.zeros([len(all_nets), len(crops)])
for idx_net, nets in enumerate(all_nets):
    tmp = np.load(PATH_TO_DATA + 'tmp_results_accuracy' + nets + '.npy')
    for idx_metric, crop_metric in enumerate([28, 24, 18, 12, 6]):
        mm[idx_net][idx_metric] = tmp[idx_metric]

cc = itertools.cycle(sns.xkcd_palette(colors))
fig, ax = pyplot.subplots()
for idx_net, nets in enumerate(['3', '8', '13', '18', '23']):
    q = np.zeros(5)
    s = np.zeros(5)
    for i, _ in enumerate(crops):
        q[i] = mm[idx_net][i]
    if idx_net ==0:
        pyplot.errorbar(x=crops, y=100*q, yerr=10*s, fmt='-o',label=name_nets[idx_net], linewidth=4)
    else:
        pyplot.errorbar(x=crops, y=100*q, yerr=10*s, fmt='-o',label=name_nets[idx_net], linewidth=4, color=next(cc))

    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

pyplot.xticks([28, 24, 20, 16, 12], ['28', '24', '20', '16', '12'])

ax.legend(loc='lower right',frameon= True, title='Regularizers')
ax.set_xlabel('Crop Size')
ax.set_ylabel('Accuracy (%)')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(25)

pyplot.gcf().subplots_adjust(bottom=0.17, top=0.96, left=0.16, right=0.96)


pyplot.savefig('./plots/accuracy/accuracy_regularizers.pdf', dpi=1000)



################################################################################################################

all_nets = [experiments.opt[i+11].name for i in range(6)]

name_nets = ['3', '9', '15', '21', '27', '32']#, '19x19', '23x23', '27x27']

mm = np.zeros([len(all_nets), len(crops)])
for idx_net, nets in enumerate(all_nets):
    tmp = np.load(PATH_TO_DATA + 'tmp_results_accuracy' + nets + '.npy')
    for idx_metric, crop_metric in enumerate([28, 24, 18, 12, 6]):
        mm[idx_net][idx_metric] = tmp[idx_metric]

cc = itertools.cycle(sns.cubehelix_palette(7, start=.5, rot=-.75))
fig, ax = pyplot.subplots()
for idx_net, nets in enumerate(all_nets):
    q = np.zeros(5)
    s = np.zeros(5)
    for i, _ in enumerate(crops):
        q[i] = mm[idx_net][i]

    if idx_net == 0:
        pyplot.errorbar(x=crops, y=100*q, yerr=10*s, fmt='-o',label=name_nets[idx_net], linewidth=4)
    else:
        pyplot.errorbar(x=crops, y=100*q, yerr=10*s, fmt='-o',label=name_nets[idx_net], linewidth=4, color=next(cc))

    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

pyplot.xticks([28, 24, 20, 16, 12], ['28', '24', '20', '16', '12'])

ax.legend(loc='upper left',frameon= True, title='Pooling Size')
ax.set_xlabel('Crop Size')
ax.set_ylabel('Accuracy (%)')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(25)

pyplot.gcf().subplots_adjust(bottom=0.17, top=0.96, left=0.16, right=0.96)


pyplot.savefig('./plots/accuracy/accuracy_pooling.pdf', dpi=1000)