import numpy as np
#import settings
import sys

import os.path

import pandas as pd
import seaborn

import itertools

import matplotlib as mpl
import seaborn as sns

import experiments_local as experiments

seaborn.set()
seaborn.set_style("whitegrid")
seaborn.set_context("poster")

import matplotlib as mpl
from matplotlib import pyplot
from matplotlib import rc
#rc('text', usetex=True)

PATH_TO_DATA = "./results/" #"/om/user/xboix/share/minimal-images/"
#"



TOTAL = 1000

crops = [28, 24, 18, 12, 6]


def get_statistics(all_nets, multi):
    mm = np.zeros([len(all_nets), 1, len(crops), 2])
    ss = np.zeros([len(all_nets), 1, len(crops), 2])
    for idx_net, nets in enumerate(all_nets):
        tmp = np.load(PATH_TO_DATA + '/tmp_results_' + multi + nets + '.npy')
        for idx_k, k in enumerate([3]):
            for idx_metric, crop_metric in enumerate(crops):
                for idx_loose, loose in enumerate([False, True]):
                    count = 0
                    for image_id in range(TOTAL):
                        #if tmp[idx_metric][idx_loose][idx_k][image_id][0] >= 0:
                        mm[idx_net][idx_k][idx_metric][idx_loose] += tmp[idx_metric][idx_loose][idx_k][image_id][0]
                        mm[idx_net][idx_k][idx_metric][idx_loose] += tmp[idx_metric][idx_loose][idx_k][image_id][1]
                        count += 1
                    mm[idx_net][idx_k][idx_metric][idx_loose] /= count

                    count = 0
                    for image_id in range(TOTAL):
                        #if tmp[idx_metric][idx_loose][idx_k][image_id][0] >= 0:
                        ss[idx_net][idx_k][idx_metric][idx_loose] += np.power(tmp[idx_metric][idx_loose][idx_k][image_id][0] - mm[idx_net][idx_k][idx_metric][idx_loose], 2)
                        ss[idx_net][idx_k][idx_metric][idx_loose] += np.power(tmp[idx_metric][idx_loose][idx_k][image_id][1] - mm[idx_net][idx_k][idx_metric][idx_loose], 2)
                        count += 1
                    ss[idx_net][idx_k][idx_metric][idx_loose] /= count
                    ss[idx_net][idx_k][idx_metric][idx_loose] = np.sqrt(ss[idx_net][idx_k][idx_metric][idx_loose])
    return mm, ss


################################################################################################

for STRICT, strict_name in enumerate(['strict', 'loose']):
    for kk in ['', 'multi']:

        all_nets = [experiments.opt[i+1].name for i in range(5)]
        name_nets = ['28 pix.', '$\geq$24 pix.',
                     '$\geq$18 pix.', '$\geq$12 pix.',
                     '$\geq$6 pix.']

        mm, ss = get_statistics(all_nets, kk)
        fig, ax = pyplot.subplots()

        cc = itertools.cycle(sns.cubehelix_palette(8))

        for idx_net, nets in enumerate(all_nets):
            q = np.zeros(len(all_nets))
            s = np.zeros(len(all_nets))
            for i, _ in enumerate(all_nets):
                q[i] = mm[idx_net][0][i][STRICT]
                s[i] = mm[idx_net][0][i][STRICT]

            if idx_net == 0:
                pyplot.errorbar(x=crops, y=100*q, yerr=10*s, fmt='-o', label=name_nets[idx_net], linewidth=4)
            else:
                pyplot.errorbar(x=crops, y=100*q, yerr=10*s, fmt='-o',
                                label=name_nets[idx_net], linewidth=4, color = next(cc))

            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

        pyplot.xticks(crops, ['28', '24', '18', '12', '6'])

        #ax.set_title('Amount of Minimal Images')
        ax.legend(loc='upper right', frameon = True, title="Crops at Training")
        ax.set_xlabel('Crop Size')
        ax.set_ylabel('% Crops Affected by a Shift')

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(25)

        pyplot.gcf().subplots_adjust(bottom=0.17, top=0.86, left=0.16, right=0.96)

        pyplot.savefig('./plots/amount_'+strict_name+'/shift_augmentation' + kk + '.pdf', dpi=1000)





        all_nets = [experiments.opt[i+6].name for i in range(5)]
        name_nets = ['Non Regularized', 'Data augment.',
                     'Dropout', 'Weight Decay',
                     'All Regularizers']

        colors = ["amber", "greyish", "orange", "black"]

        mm, ss = get_statistics(all_nets, kk)
        fig, ax = pyplot.subplots()

        cc = itertools.cycle(sns.xkcd_palette(colors))

        for idx_net, nets in enumerate(all_nets):
            q = np.zeros(len(all_nets))
            s = np.zeros(len(all_nets))
            for i, _ in enumerate(all_nets):
                q[i] = mm[idx_net][0][i][STRICT]
                s[i] = mm[idx_net][0][i][STRICT]

            if idx_net == 0:
                pyplot.errorbar(x=crops, y=100*q, yerr=10*s, fmt='-o', label=name_nets[idx_net], linewidth=4)
            else:
                pyplot.errorbar(x=crops, y=100*q, yerr=10*s, fmt='-o',
                                label=name_nets[idx_net], linewidth=4, color = next(cc))

            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

        pyplot.xticks(crops, ['28', '24', '18', '12', '6'])

        #ax.set_title('Amount of Minimal Images')
        ax.legend(loc='upper right', frameon = True, title="Regularizer")
        ax.set_xlabel('Crop Size')
        ax.set_ylabel('% Crops Affected by a Shift')

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(25)

        pyplot.gcf().subplots_adjust(bottom=0.17, top=0.86, left=0.16, right=0.96)

        pyplot.savefig('./plots/amount_'+strict_name+'/shift_regularizers' + kk + '.pdf', dpi=1000)






        all_nets = [experiments.opt[i+11].name for i in range(3)]
        name_nets = ['3x3', '7x7',
                     '13x13']

        mm, ss = get_statistics(all_nets, kk)
        fig, ax = pyplot.subplots()

        cc = itertools.cycle(sns.cubehelix_palette(3, start=.5, rot=-.75))

        for idx_net, nets in enumerate(all_nets):
            q = np.zeros(len(crops))
            s = np.zeros(len(crops))
            for i, _ in enumerate(crops):
                q[i] = mm[idx_net][0][i][STRICT]
                s[i] = mm[idx_net][0][i][STRICT]

            if idx_net == 0:
                pyplot.errorbar(x=crops, y=100*q, yerr=10*s, fmt='-o', label=name_nets[idx_net], linewidth=4)
            else:
                pyplot.errorbar(x=crops, y=100*q, yerr=10*s, fmt='-o',
                                label=name_nets[idx_net], linewidth=4, color = next(cc))

            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

        pyplot.xticks(crops, ['28', '24', '18', '12', '6'])

        #ax.set_title('Amount of Minimal Images')
        ax.legend(loc='upper right', frameon = True, title="Pooling Size")
        ax.set_xlabel('Crop Size')
        ax.set_ylabel('% Crops Affected by a Shift')

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(25)

        pyplot.gcf().subplots_adjust(bottom=0.17, top=0.86, left=0.16, right=0.96)

        pyplot.savefig('./plots/amount_'+strict_name+'/shift_pooling' + kk + '.pdf', dpi=1000)


################################################################################################
################################################################################################