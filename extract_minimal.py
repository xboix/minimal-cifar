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

TOTAL = 1000

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

crop_size = int(sys.argv[1:][1])
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


patches = tf.extract_image_patches(
    images=images_in,
    ksizes=[1, experiments.crop_sizes[crop_size], experiments.crop_sizes[crop_size], 1],
    strides=[1, 1, 1, 1],
    rates=[1, 1, 1, 1],
    padding='VALID')

map_size = (opt.hyper.image_size - experiments.crop_sizes[crop_size] + 1)

patches = tf.reshape(patches, [-1, experiments.crop_sizes[crop_size], experiments.crop_sizes[crop_size], 3])

patches = tf.image.resize_nearest_neighbor(patches, [opt.hyper.image_size, opt.hyper.image_size])

ims = tf.unstack(patches, num=map_size**2, axis=0)
process_ims = []
for im in ims: #Get each individual image
    imc = tf.image.per_image_standardization(im)
    imc.set_shape([opt.hyper.image_size, opt.hyper.image_size, 3])
    process_ims.append(imc)

image = tf.stack(process_ims)

if opt.extense_summary:
    tf.summary.image('input', image)

# Call DNN
dropout_rate = tf.placeholder(tf.float32)
to_call = getattr(nets, opt.dnn.name)
y, parameters, _ = to_call(image, dropout_rate, opt, dataset.list_labels)


# Accuracy
with tf.name_scope('accuracy'):
    top_class = tf.argmax(y, 1)
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)
################################################################################################

if not os.path.exists(opt.log_dir_base + opt.name + '/maps'):
    os.makedirs(opt.log_dir_base + opt.name + '/maps')
    os.makedirs(opt.log_dir_base + opt.name + '/maps/top/')
    os.makedirs(opt.log_dir_base + opt.name + '/maps/confidence/')

if not os.path.exists(opt.log_dir_base + opt.name + '/maps/top/' + str(experiments.crop_sizes[crop_size])):
    os.makedirs(opt.log_dir_base + opt.name + '/maps/top/' + str(experiments.crop_sizes[crop_size]))
    os.makedirs(opt.log_dir_base + opt.name + '/maps/confidence/' + str(experiments.crop_sizes[crop_size]))

with tf.Session() as sess:

    ################################################################################################
    # Set up checkpoints and data
    ################################################################################################

    saver = tf.train.Saver(max_to_keep=opt.max_to_keep_checkpoints)

    # Automatic restore model, or force train from scratch
    flag_testable = False

    # Set up directories and checkpoints
    if not os.path.isfile(opt.log_dir_base + opt.name + '/models/checkpoint'):
        sess.run(tf.global_variables_initializer())
    else:
        print("RESTORE")
        saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/models/'))
        flag_testable = True

    ################################################################################################

    ################################################################################################
    # RUN TEST
    ################################################################################################

    if flag_testable:

        test_handle_full = sess.run(test_iterator_full.string_handle())
        # Run one pass over a batch of the test dataset.
        sess.run(test_iterator_full.initializer)

        pred_map = np.zeros([TOTAL, map_size, map_size])
        top_map = np.zeros([TOTAL, map_size, map_size])
        pred_multi = np.zeros([TOTAL, map_size, map_size])

        for num_iter in range(TOTAL):

            top_map, gt, pred_map, top_multi_map = sess.run([y, y_, correct_prediction, top_class], feed_dict={handle: test_handle_full,
                                                      dropout_rate: opt.hyper.drop_test})

            a = np.reshape(pred_map,[map_size, map_size])
            print(np.shape(a))
            print(pred_map[num_iter, :, :].shape)

            pred_map[num_iter, :, :] = a
            top_map[num_iter, :, :] = np.reshape(top_map[:, gt[0]], [map_size, map_size])
            top_multi_map[num_iter, :, :] = np.reshape(top_multi_map, [map_size, map_size])
            print(num_iter)
            sys.stdout.flush()

        with open(opt.log_dir_base + opt.name + '/maps/top/' + str(experiments.crop_sizes[crop_size])
                  + '/maps.pkl', 'wb') as f:
            pickle.dump(pred_map, f)

        with open(opt.log_dir_base + opt.name + '/maps/confidence/' + str(experiments.crop_sizes[crop_size])
                 + '/maps.pkl', 'wb') as f:
            pickle.dump(top_map, f)

        with open(opt.log_dir_base + opt.name + '/maps/top_multi/' + str(experiments.crop_sizes[crop_size])
                 + '/maps.pkl', 'wb') as f:
            pickle.dump(top_map, f)

    else:
        print("MODEL WAS NOT TRAINED")

print(":)")