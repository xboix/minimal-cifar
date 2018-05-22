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
rc('text', usetex=True)

PATH_TO_DATA = "./results/" #"/om/user/xboix/share/minimal-images/"
#"


mm = np.zeros([5, 5,  5, 2])
ss = np.zeros([5, 5,  5, 2])
TOTAL = 1000


all_nets = [experiments.opt[i+11].name for i in range(3)]

for idx_net, nets in enumerate(all_nets):
    tmp = np.load(PATH_TO_DATA + '/tmp_results_multi' + nets + '.npy')
    for idx_k, k in enumerate([3]):
        for idx_metric, crop_metric in enumerate([28, 24, 18, 12, 6]):
            for idx_loose, loose in enumerate([False, True]):
                count = 0
                for image_id in range(TOTAL):
                    if tmp[idx_metric][idx_loose][idx_k][image_id][0] >= 0:
                        mm[idx_net][idx_k][idx_metric][idx_loose] += tmp[idx_metric][idx_loose][idx_k][image_id][0]
                        mm[idx_net][idx_k][idx_metric][idx_loose] += tmp[idx_metric][idx_loose][idx_k][image_id][1]
                        count += 1
                mm[idx_net][idx_k][idx_metric][idx_loose] /= count

                count = 0
                for image_id in range(500):
                    if tmp[idx_metric][idx_loose][idx_k][image_id][0] >= 0:
                        ss[idx_net][idx_k][idx_metric][idx_loose] += np.power(tmp[idx_metric][idx_loose][idx_k][image_id][0] - mm[idx_net][idx_k][idx_metric][idx_loose], 2)
                        ss[idx_net][idx_k][idx_metric][idx_loose] += np.power(tmp[idx_metric][idx_loose][idx_k][image_id][1] - mm[idx_net][idx_k][idx_metric][idx_loose], 2)
                        count += 1
                ss[idx_net][idx_k][idx_metric][idx_loose] /= count
                ss[idx_net][idx_k][idx_metric][idx_loose] = np.sqrt(ss[idx_net][idx_k][idx_metric][idx_loose])

''' 
fig, ax = pyplot.subplots()
for idx_net, nets in enumerate(['vgg16', 'resnet', 'inception']):
    q = np.zeros(5)
    s = np.zeros(5)
    for i, _ in enumerate([3, 5, 7, 11, 17]):
        q[i] = mm[idx_net+1][i][0][1]
        s[i] = ss[idx_net+1][i][0][1]
    ax.errorbar(x=[3, 5, 7, 11, 17], y=100*q, yerr=10*s, label=nets,linewidth=4, fmt='-o')
    #ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_title('ImageNet\n Fluctuation of the Accuracy due to FoV Shifts')
ax.legend(loc='upper left',  frameon= True)
ax.set_xlabel('Size of the FoV Shift (Pixels)')
ax.set_ylabel('\% of Crops Sensitive to FoV Shifts')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(25)

pyplot.gcf().subplots_adjust(bottom=0.17, top=0.86, left=0.16, right=0.96)

pyplot.savefig('./area.pdf', dpi=1000)
'''

fig, ax = pyplot.subplots()
for idx_net, nets in enumerate(['3', '8', '13', '18', '23']):
    q = np.zeros(5)
    s = np.zeros(5)
    for i, _ in enumerate([28, 24, 18, 12, 6]):
        q[i] = mm[idx_net][0][i][1]
        s[i] = mm[idx_net][0][i][1]
    pyplot.errorbar(x=[28, 24, 18, 12, 6], y=100*q, yerr=10*s, fmt='-o',label=nets, linewidth=4)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

pyplot.xticks([28, 24, 18, 12, 6], ['28', '24', '18', '12', '6'])

ax.set_title('ImageNet\n Amount of ``Strict\'\' Minimal Images')
ax.legend(loc='upper right',frameon= True)
ax.set_xlabel('Crop Scale Factor')
ax.set_ylabel('\% of Crops Sensitive to 1 Pixel Shifts')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(25)

pyplot.gcf().subplots_adjust(bottom=0.17, top=0.86, left=0.16, right=0.96)


pyplot.savefig('./scale_strict.pdf', dpi=1000)


mm = np.zeros([4, 10, 4, 2])
for idx_net, nets in enumerate(['alexnet', 'vgg16', 'resnet', 'inception']):
    tmp = np.load(PATH_TO_DATA + 'area/tmp_results_' + nets + '.npy')
    for idx_k, k in enumerate([3]):
        for idx_metric, crop_metric in enumerate([0.2, 0.4, 0.6, 0.8]):
            for idx_loose, loose in enumerate([False, True]):
                count_im = 0
                for image_cat in range(10):
                    count = 0
                    for image_id in range(50):
                        if tmp[idx_metric][idx_loose][idx_k][count_im][0] >= 0:
                            mm[idx_net][image_cat][idx_metric][idx_loose] += tmp[idx_metric][idx_loose][idx_k][count_im][0]
                            mm[idx_net][image_cat][idx_metric][idx_loose] += tmp[idx_metric][idx_loose][idx_k][count_im][1]
                            count += 1
                        count_im += 1
                    mm[idx_net][image_cat][idx_metric][idx_loose] /= count

print(mm)
cc=['dog', 'snake', 'monkey', 'fish', 'vegetable', 'instrument', 'boat', 'vehicle', 'drinks', 'furniture']
fig, ax = pyplot.subplots()
q = np.zeros([10, 3])
for idx_net, nets in enumerate(['vgg16', 'resnet', 'inception']):
    for i in range(10):
        q[i][idx_net] = mm[idx_net+1][i][0][0]

opacity = 1
bar_width = 0.2


rects1 = pyplot.bar(np.arange(0,10)+0*bar_width, 100*np.squeeze(q[:,0]), bar_width,
                 alpha=opacity,
                 color=seaborn.color_palette()[0],
                 label='vgg16')

rects3 = pyplot.bar(np.arange(0,10)+1*bar_width, 100*np.squeeze(q[:,2]),bar_width,
                 alpha=opacity,
                 color=seaborn.color_palette()[1],
                 label='inception')

rects2 = pyplot.bar(np.arange(0,10)+2*bar_width, 100*np.squeeze(q[:,1]), bar_width,
                 alpha=opacity,
                 color=seaborn.color_palette()[2],
                 label='resnet')



pyplot.xticks(rotation=45)
ax.set_xticks([0,1,1.8,3.1,3.8,4.7,6,7,8,9])
ax.set_xticklabels(cc)

ax.legend(loc='upper center', frameon= True)
ax.set_title('ImageNet\n Amount of ``Strict\'\' minimal images per class')
ax.set_ylabel('\% of Crops Sensitive to 1 Pixel Shifts')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(25)
ax.grid(False)

# Turns off grid on the secondary (right) Axis.
pyplot.gcf().subplots_adjust(bottom=0.23, top=0.83, left=0.16, right=0.96)
pyplot.savefig('./classes_strict.pdf', dpi=1000)


mm_pos = np.zeros([4, 5, 4, 2])
mm_neg = np.zeros([4, 5, 4, 2])

for idx_net, nets in enumerate(all_nets):
    tmp = np.load(PATH_TO_DATA + 'tmp_results_' + nets + '.npy')
    for idx_k, k in enumerate([3, 5, 7, 11, 17]):
        for idx_metric, crop_metric in enumerate([0.2, 0.4, 0.6, 0.8]):
            for idx_loose, loose in enumerate([False, True]):
                count = 0
                for image_id in range(500):
                    if tmp[idx_metric][idx_loose][idx_k][image_id][0] >= 0:
                        mm_pos[idx_net][idx_k][idx_metric][idx_loose] += tmp[idx_metric][idx_loose][idx_k][image_id][0]
                        mm_neg[idx_net][idx_k][idx_metric][idx_loose] += tmp[idx_metric][idx_loose][idx_k][image_id][1]
                        count += 1
                mm_pos[idx_net][idx_k][idx_metric][idx_loose] /= count
                mm_neg[idx_net][idx_k][idx_metric][idx_loose] /= count


fig, ax = pyplot.subplots()
colors = ['b','g','r']
for idx_net, nets in enumerate(['vgg16', 'resnet', 'inception']):
    q = np.zeros(4)
    for i, _ in enumerate([0.2, 0.4, 0.6, 0.8]):
        q[i] = mm_pos[idx_net+1][0][i][0]
    pyplot.errorbar(x=[0.2, 0.4, 0.6, 0.8], y=100*q,  color=colors[idx_net],fmt=':o',label=nets + ' Positive', linewidth=4)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    for i, _ in enumerate([0.2, 0.4, 0.6, 0.8]):
        q[i] = mm_neg[idx_net+1][0][i][0]
    pyplot.errorbar(x=[0.2, 0.4, 0.6, 0.8], y=100*q,  color=colors[idx_net],fmt='--o',label=nets + ' Negative', linewidth=4)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

pyplot.xticks([0.2, 0.4, 0.6, 0.8], ['1/5', '2/5', '3/5', '4/5'])
ax.set_title('ImageNet\n Amount of ``Strict\'\' Minimal Images')
ax.legend(loc='upper right', frameon= True)
ax.set_xlabel('Crop Relative Scale')
ax.set_ylabel('\% of Crops Sensitive to 1 Pixel Shifts')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(25)

pyplot.gcf().subplots_adjust(bottom=0.17, top=0.86, left=0.16, right=0.96)

pyplot.savefig('./scale_posneg_strict.pdf', dpi=1000)


mm = np.zeros([4, 4, 2])

image_scale = '1.0'


for idx_net, nets in enumerate(all_nets):
    tmp = np.load(PATH_TO_DATA + 'tmp_results_' + nets + '_small.npy')

    for idx_metric, crop_metric in enumerate([0.2, 0.4, 0.6, 0.8]):
        for idx_loose, loose in enumerate([False, True]):
            count = 0
            for image_id in range(500):
                if tmp[idx_metric][idx_loose][image_id][0] >= 0:
                    mm[idx_net][idx_metric][idx_loose] += tmp[idx_metric][idx_loose][image_id][0]
                    mm[idx_net][idx_metric][idx_loose] += tmp[idx_metric][idx_loose][image_id][1]
                    count += 1
            mm[idx_net][idx_metric][idx_loose] /= count

print(mm)


fig, ax = pyplot.subplots()
for idx_net, nets in enumerate(['vgg16', 'resnet', 'inception']):
    q = np.zeros(4)
    s = np.zeros(4)
    for i, _ in enumerate([0.2, 0.4, 0.6, 0.8]):
        q[i] = mm[idx_net+1][i][0]
        s[i] = mm[idx_net+1][i][0]
    pyplot.errorbar(x=[0.2, 0.4, 0.6, 0.8], y=100*q, yerr=10*s, fmt='-o',label=nets, linewidth=2, capsize=20)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

ax.set_title('ImageNet\n Amount of ``Strict\'\' Minimal Images')
ax.legend(loc='upper right', frameon= True)
ax.set_xlabel('Crop Relative Scale')
ax.set_ylabel('\% of Crops Sensitive to 2 Pixel Shrink')
pyplot.xticks([0.2, 0.4, 0.6, 0.8], ['1/5', '2/5', '3/5', '4/5'])

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(25)
pyplot.xticks([0.2, 0.4, 0.6, 0.8], ['1/5', '2/5', '3/5', '4/5'])

pyplot.gcf().subplots_adjust(bottom=0.17, top=0.86, left=0.16, right=0.96)
pyplot.savefig('./scale_strict_small.pdf', dpi=1000)


mm = np.zeros([4, 4, 2, 10])

for idx_net, nets in enumerate(all_nets):
    tmp = np.load(PATH_TO_DATA + 'tmp_results_' + nets + '_small.npy')

    for idx_metric, crop_metric in enumerate([0.2, 0.4, 0.6, 0.8]):
        for idx_loose, loose in enumerate([False, True]):
            count_im = 0
            for image_cat in range(10):
                count = 0
                for image_id in range(50):
                    if tmp[idx_metric][idx_loose][count_im][0] >= 0:
                        mm[idx_net][idx_metric][idx_loose][image_cat] += tmp[idx_metric][idx_loose][count_im][0]
                        mm[idx_net][idx_metric][idx_loose][image_cat] += tmp[idx_metric][idx_loose][count_im][1]
                        count += 1
                    count_im += 1
                mm[idx_net][idx_metric][idx_loose][image_cat] /= count

print(mm)

