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
import matplotlib.ticker as mtick

import itertools


import matplotlib as mpl
from matplotlib import pyplot
from matplotlib import rc
#rc('text', usetex=True)

import seaborn as sns

PATH_TO_DATA = "./results/" #"/om/user/xboix/share/minimal-images/"
#"

TOTAL = 1000

crops = [28, 24, 18, 12, 6]
for STRICT, strict_name in enumerate(['strict', 'loose']):
    for kk in ['', 'multi']:

        all_nets = [experiments.opt[i+1].name for i in range(5)]
        name_nets = ['28 pix.', '$\geq$24 pix.',
                     '$\geq$18 pix.', '$\geq$12 pix.',
                     '$\geq$6 pix.']

        for idx_metric, crop_metric in enumerate(crops):
            cc = itertools.cycle(sns.cubehelix_palette(8))

            fig, ax = pyplot.subplots()
            for idx_net, nets in enumerate(all_nets):
                tmp = np.load(PATH_TO_DATA + '/tmp_results_' + kk + nets + '.npy')
                mm = np.zeros([TOTAL])
                for image_id in range(TOTAL):
                    mm[image_id] += tmp[idx_metric][STRICT][0][image_id][0]
                    mm[image_id] += tmp[idx_metric][STRICT][0][image_id][1]

                if idx_net == 0:
                    sns.kdeplot(100*mm, clip=(0.0, 100.0), label=name_nets[idx_net], linewidth=4)
                else:
                    sns.kdeplot(100*mm, clip=(0.0, 100.0), label=name_nets[idx_net], linewidth=4, color = next(cc))

            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            ax.ticklabel_format(useOffset=True)

            mf = mpl.ticker.ScalarFormatter(useMathText=True)
            mf.set_powerlimits((-2, 10))
            pyplot.gca().yaxis.set_major_formatter(mf)

            ax.set_title(str(crops[idx_metric]) + 'pix. Min. Images')

            ax.set_xlabel('% of Crops Sensitive to 1 Pixel Shifts')
            ax.set_ylabel('Probability Density')
            if idx_metric>1:
                loc = 'upper left'
            else:
                loc = 'upper right'
            ax.legend(loc=loc, frameon=True, title="Crops at Training")

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(25)
            if STRICT:
                pyplot.xlim((0, 100.0))

            pyplot.gcf().subplots_adjust(bottom=0.17, top=0.90, left=0.20, right=0.96)

            pyplot.savefig('./plots/distributions_'+strict_name+'/distribution_augmentation_' + str(crop_metric) + kk + '.pdf', dpi=1000)



        all_nets = [experiments.opt[i+6].name for i in range(5)]
        name_nets = ['Non Regularized', 'Data augment.',
                     'Dropout', 'Weight Decay',
                     'All Regularizers']

        colors = ["amber", "greyish", "orange", "black"]

        for idx_metric, crop_metric in enumerate(crops):
            cc = itertools.cycle(sns.xkcd_palette(colors))
            fig, ax = pyplot.subplots()
            for idx_net, nets in enumerate(all_nets):

                tmp = np.load(PATH_TO_DATA + '/tmp_results_'+ kk + nets + '.npy')
                mm = np.zeros([TOTAL])
                for image_id in range(TOTAL):
                    mm[image_id] += tmp[idx_metric][STRICT][0][image_id][0]
                    mm[image_id] += tmp[idx_metric][STRICT][0][image_id][1]

                if idx_net == 0:
                    sns.kdeplot(100*mm, clip=(0.0, 100.0), label=name_nets[idx_net], linewidth=4)
                else:
                    sns.kdeplot(100*mm, clip=(0.0, 100.0), label=name_nets[idx_net], linewidth=4, color = next(cc))

            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            ax.ticklabel_format(useOffset=True)

            mf = mpl.ticker.ScalarFormatter(useMathText=True)
            mf.set_powerlimits((-2, 10))
            pyplot.gca().yaxis.set_major_formatter(mf)

            ax.set_title(  str(crops[idx_metric]) + 'pix. Minimal Images')
            ax.set_xlabel('% of Crops Sensitive to 1 Pixel Shifts')
            ax.set_ylabel('Probability Density')

            if idx_metric>1:
                loc = 'upper left'
            else:
                loc = 'upper right'

            ax.legend(loc=loc, frameon=True, title="Regularizers")

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(25)
            if STRICT:
                pyplot.xlim((0, 100.0))

            pyplot.gcf().subplots_adjust(bottom=0.17, top=0.85, left=0.20, right=0.96)

            pyplot.savefig('./plots/distributions_'+strict_name+'/distribution_regularizers_' + str(crop_metric) + kk + '.pdf', dpi=1000)




        all_nets = [experiments.opt[i+11].name for i in range(3)]
        name_nets = ['3x3', '7x7',
                     '13x13']

        for idx_metric, crop_metric in enumerate(crops):
            cc = itertools.cycle(sns.cubehelix_palette(3, start=.5, rot=-.75))

            fig, ax = pyplot.subplots()
            for idx_net, nets in enumerate(all_nets):
                tmp = np.load(PATH_TO_DATA + '/tmp_results_' + kk + nets + '.npy')
                mm = np.zeros([TOTAL])
                for image_id in range(TOTAL):
                    mm[image_id] += tmp[idx_metric][STRICT][0][image_id][0]
                    mm[image_id] += tmp[idx_metric][STRICT][0][image_id][1]

                if idx_net == 0:
                    sns.kdeplot(100*mm, clip=(0.0, 100.0), label=name_nets[idx_net], linewidth=4)
                else:
                    sns.kdeplot(100*mm, clip=(0.0, 100.0), label=name_nets[idx_net], linewidth=4, color = next(cc))

            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            ax.ticklabel_format(useOffset=True)

            mf = mpl.ticker.ScalarFormatter(useMathText=True)
            mf.set_powerlimits((-2, 10))
            pyplot.gca().yaxis.set_major_formatter(mf)

            ax.set_title( str(crops[idx_metric]) + 'pix. Minimal Images')
            ax.set_xlabel('% of Crops Sensitive to 1 Pixel Shifts')
            ax.set_ylabel('Probability Density')
            if idx_metric>1:
                loc = 'upper left'
            else:
                loc = 'upper right'
            ax.legend(loc = loc, frameon=True, title="Pooling Size")

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(25)
            if STRICT:
                pyplot.xlim((0, 100.0))

            pyplot.gcf().subplots_adjust(bottom=0.17, top=0.85, left=0.20, right=0.96)

            pyplot.savefig('./plots/distributions_'+strict_name+'/distribution_pooling_' + str(crop_metric) + kk + '.pdf', dpi=1000)

