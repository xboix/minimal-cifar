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

TOTAL = 10000

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
print('IMAGES_IN SHAPE:', images_in.get_shape())
images_in_exp = tf.tile(images_in, tf.constant([

# patches = tf.extract_image_patches(
#     images=images_in,
#     ksizes=[1, experiments.crop_sizes[crop_size], experiments.crop_sizes[crop_size], 1],
#     strides=[1, 1, 1, 1],
#     rates=[1, 1, 1, 1],
#     padding='VALID')

mask = tf.ones([1, experiments.crop_sizes[crop_size], experiments.crop_sizes[crop_size], 3])
masks = []
total_pad = opt.hyper.image_size - experiments.crop_sizes[crop_size]
for i in range(total_pad + 1):
    for j in range(total_pad + 1):
        full_mask = tf.pad(mask, tf.constant([i, total_pad - i], [j, total_pad - j]))
        full_masks.append(full_mask)
masks = tf.stack(masks)
images_in_exp = tf.tile(images_in, tf.constant([masks.get_shape()[0], 1, 1, 1]))
masked_ims = tf.multiply(images_in, masks)	# mask each tiled image with one of the masks 

# ims = tf.unstack(patches, num=map_size**2, axis=0)
ims = tf.unstack(masked_ims)
process_ims = []
eccentricity_test = False	# True if testing with eccentricity
ecc_crop_size = 20
for im in ims: #Get each individual image
    if eccentricity_test:
       imc_small = tf.image.resize_images(im, [ecc_crop_size, ecc_crop_size])
       imc_crop = tf.image.central_crop(im, float(ecc_crop_size) / opt.hyper.image_size)
       im = tf.concat([imc_small, imc_crop], 2) 
    imc = tf.image.per_image_standardization(im)
    if eccentricity_test:
        imc.set_shape([ecc_crop_size, ecc_crop_size, 6])
    else:
        imc.set_shape([opt.hyper.image_size, opt.hyper.image_size, 3])
    process_ims.append(imc)

image = tf.stack(process_ims)

# if opt.extense_summary:
#     tf.summary.image('input', image)

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
    os.makedirs(opt.log_dir_base + opt.name + '/maps/top_multi/')


if not os.path.exists(opt.log_dir_base + opt.name + '/maps/top/' + str(experiments.crop_sizes[crop_size])):
    os.makedirs(opt.log_dir_base + opt.name + '/maps/top/' + str(experiments.crop_sizes[crop_size]))
    os.makedirs(opt.log_dir_base + opt.name + '/maps/confidence/' + str(experiments.crop_sizes[crop_size]))
    os.makedirs(opt.log_dir_base + opt.name + '/maps/top_multi/' + str(experiments.crop_sizes[crop_size]))


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

        pred_map_total = np.zeros([TOTAL, map_size, map_size])
        top_map_total = np.zeros([TOTAL, map_size, map_size])
        top_multi_total = np.zeros([TOTAL, map_size, map_size])

        for num_iter in range(TOTAL):

            top_map, gt, pred_map, top_multi_map = sess.run([y, y_, correct_prediction, top_class], feed_dict={handle: test_handle_full,
                                                      dropout_rate: opt.hyper.drop_test})

            pred_map_total[num_iter, :, :] = np.reshape(pred_map,[map_size, map_size])
            top_map_total[num_iter, :, :] = np.reshape(top_map[:, gt[0]], [map_size, map_size])
            top_multi_total[num_iter, :, :] = np.reshape(top_multi_map, [map_size, map_size])
            if not num_iter % 1000:
                print(num_iter)
            sys.stdout.flush()

        with open(opt.log_dir_base + opt.name + '/maps/top/' + str(experiments.crop_sizes[crop_size])
                  + '/maps_pad.pkl', 'wb') as f:
            pickle.dump(pred_map_total, f)

        with open(opt.log_dir_base + opt.name + '/maps/confidence/' + str(experiments.crop_sizes[crop_size])
                 + '/maps_pad.pkl', 'wb') as f:
            pickle.dump(top_map_total, f)

        with open(opt.log_dir_base + opt.name + '/maps/top_multi/' + str(experiments.crop_sizes[crop_size])
                 + '/maps_pad.pkl', 'wb') as f:
            pickle.dump(top_multi_total, f)

    else:
        print("MODEL WAS NOT TRAINED")

print(":)")
