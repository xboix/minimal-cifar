import os.path
import shutil
import sys
import numpy as np

import tensorflow as tf

import experiments
from datasets import cifar_dataset
from nets import nets
from util import summary
import pickle

import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['ps.useafm'] = True
mpl.rcParams.update({'font.size': 14})
mpl.rc('axes', labelsize=14)
mpl.rc('ytick', labelsize=14)
mpl.rc('xtick', labelsize=14)

from pylab import *


TOTAL = 100

################################################################################################
# Read experiment to run
################################################################################################

ID = int(sys.argv[1:][0])

opt = experiments.opt[ID]

# Skip execution if instructed in experiment
if opt.skip:
    print("SKIP")
    quit()

print(opt.name)

################################################################################################

################################################################################################
# Define training and validation datasets through Dataset API
################################################################################################

opt.hyper.batch_size = 1

# Initialize dataset and creates TF records if they do not exist
dataset = cifar_dataset.Cifar10(opt)

# No repeatable dataset for testing
test_dataset_full = dataset.create_dataset(augmentation=False, standarization=False, set_name='test', repeat=False)

# Hadles to switch datasets
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.contrib.data.Iterator.from_string_handle(
    handle, test_dataset_full.output_types, test_dataset_full.output_shapes)

test_iterator_full = test_dataset_full.make_initializable_iterator()
################################################################################################


################################################################################################
# Declare DNN
################################################################################################

# Get data from dataset dataset
images_in, y_ = iterator.get_next()


if not os.path.exists(opt.log_dir_base + '/../cifar-images'):
    os.makedirs(opt.log_dir_base + '/../cifar-images')


with tf.Session() as sess:
    test_handle_full = sess.run(test_iterator_full.string_handle())
    # Run one pass over a batch of the test dataset.

    sess.run(test_iterator_full.initializer)
    for num_iter in range(TOTAL):
        im, = sess.run([images_in], feed_dict={handle: test_handle_full})

        im = np.squeeze(im).astype("uint8")
        im = np.reshape(im , (32, 32, 3), order='F')
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.imshow(im)
        ax.set_axis_off()
        plt.savefig(opt.log_dir_base + '/../cifar-images/'
                    + str(num_iter) + '.pdf', format='pdf', dpi=1000)

        plt.close('all')
        print(num_iter)
        sys.stdout.flush()

print(":)")

