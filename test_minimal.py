import os.path
import shutil
import sys
import numpy as np

import tensorflow as tf

import experiments
from datasets import cifar_dataset
from nets import nets
from util import summary


################################################################################################
# Read experiment to run
################################################################################################

print('entered test_minimal')

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
print('Defining datasets')
################################################################################################

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
print('Declaring DNN')
################################################################################################

# Get data from dataset dataset
images_in, y_ = iterator.get_next()


def get_candidates(im):
    candidate_transformations = [
        lambda: tf.random_crop(im, [experiments.crop_sizes[0], experiments.crop_sizes[0], 3]),
        lambda: tf.random_crop(im, [experiments.crop_sizes[1], experiments.crop_sizes[1], 3]),
        lambda: tf.random_crop(im, [experiments.crop_sizes[2], experiments.crop_sizes[2], 3]),
        lambda: tf.random_crop(im, [experiments.crop_sizes[3], experiments.crop_sizes[3], 3]),
        lambda: tf.random_crop(im, [experiments.crop_sizes[4], experiments.crop_sizes[4], 3])
    ]
    return candidate_transformations


def aux_transf(im, rr):

    candidate_transformations = get_candidates(im)

    pred_fn_pairs = []
    pred_fn_pairs.append((tf.equal(rr, tf.constant(0)), candidate_transformations[0]))

    cc = 1
    for t in range(len(candidate_transformations) - 1):
        pred_fn_pairs.append((
            tf.equal(rr, tf.constant(cc)),
            candidate_transformations[t]))
        cc += 1

    return pred_fn_pairs


crop_size = tf.placeholder(tf.int32)

ims = tf.unstack(images_in, num=opt.hyper.batch_size, axis=0)

eccentricity_test = True
ecc_crop_size = 20
process_ims = []
for im in ims: #Get each individual image
    imc = tf.case(pred_fn_pairs=aux_transf(im, crop_size), default=lambda: 0*im)
    imc = tf.image.resize_images(imc, [opt.hyper.image_size, opt.hyper.image_size])
    if eccentricity_test:
        imc_small = tf.image.resize_images(imc, [ecc_crop_size, ecc_crop_size])
        imc_crop = tf.image.central_crop(imc, float(ecc_crop_size) / opt.hyper.image_size)
        imc = tf.concat([imc_small, imc_crop], 2) 

    imc = tf.image.per_image_standardization(imc)
#     imc.set_shape([opt.hyper.image_size, opt.hyper.image_size, 3])
    process_ims.append(imc)

image = tf.stack(process_ims)

# image.set_shape([opt.hyper.batch_size, opt.hyper.image_size, opt.hyper.image_size, 3])

# if opt.extense_summary:
#     tf.summary.image('input', image)

# Call DNN
print('Calling DNN')
dropout_rate = tf.placeholder(tf.float32)
to_call = getattr(nets, opt.dnn.name)
y, parameters, _ = to_call(image, dropout_rate, opt, dataset.list_labels)

# Accuracy
print('Getting accuracy')
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)
################################################################################################


with tf.Session() as sess:

    ################################################################################################
    # Set up checkpoints and data
    print('Setting up checkpoints')
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
    print('Running test')
    ################################################################################################

    results = np.zeros([len(experiments.crop_sizes)])

    if flag_testable:

        test_handle_full = sess.run(test_iterator_full.string_handle())
        for cc in range(len(experiments.crop_sizes)):
            # Run one pass over a batch of the test dataset.
            sess.run(test_iterator_full.initializer)
            acc_tmp = 0.0
            for num_iter in range(int(dataset.num_images_test / opt.hyper.batch_size)):
                acc_val = sess.run([accuracy], feed_dict={handle: test_handle_full,
                                                          dropout_rate: opt.hyper.drop_test,
                                                          crop_size: cc})
                acc_tmp += acc_val[0]

            val_acc = acc_tmp / float(int(dataset.num_images_test / opt.hyper.batch_size))
            print("Full test acc for crop_size=" + str(experiments.crop_sizes[cc]) + ": " + str(val_acc))
            sys.stdout.flush()
            results[cc] = val_acc
        print(":)")

    else:
        print("MODEL WAS NOT TRAINED")

print('Saving')
np.save(opt.log_dir_base + opt.name + '/tmp_results_accuracy' + opt.name + '.npy', results)
print(':)')
