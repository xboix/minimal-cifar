import numpy as np
import sys
import experiments as experiments
import pickle

import os.path

ID = int(sys.argv[1:][0])

opt = experiments.opt[ID]


def create_location_minimal_image_maps_multi(image_id, top5map_general, top5map_general_small, loose, k=1):

    l_top5_general = top5map_general[image_id, :, :]
    s_top5_general = top5map_general_small[image_id, :, :]

    r, c = l_top5_general.shape
    M = np.zeros((r, c))

    for ll in np.unique(l_top5_general):

        l_top5 = np.copy(l_top5_general)
        l_top5[l_top5_general == ll] = 1
        l_top5[l_top5_general != ll] = 0

        s_top5 = np.copy(s_top5_general)
        s_top5[s_top5_general == ll] = 1
        s_top5[s_top5_general != ll] = 0

        for i in range(r):
            for j in range(c):
                self = l_top5[i, j]

                window = s_top5[i:i + 3, j:j + 3]  # get all the possible shrinks for this crop
                if loose:
                    if self:  # if the current crop is correctly classified...
                        if not np.all(window):  # if any cell in the window is incorrectly classified...
                            M[
                                i, j] = 1.  # ...the current crop is a positive minimal image. Otherwise, it's not minimal.
                    else:  # if the current crop is incorrectly classified...
                        if np.any(window):  # if any cell in the window is correctly classified...
                            M[
                                i, j] = -1.  # ...the current crop is a negative minimal image. Otherwise, it's not minimal.
                else:  # we are looking for strict minimal image maps
                    if self:  # if the current crop is correctly classified...
                        if not np.any(window):  # if all crops in the window are incorrectly classified...
                            M[
                                i, j] = 1.  # ...the current crop is a positive minimal image. Otherwise, it's not minimal.
                    else:  # if the current crop is incorrectly classified...
                        if np.all(window):  # if all the crops in the window are correctly classified...
                            M[
                                i, j] = -1.  # ...the current crop is a negative minimal image. Otherwise, it's not minimal.

    # calculate map statistics
    num_pos_min_imgs = (M > 0.).sum()
    num_neg_min_imgs = (M < 0.).sum()

    return num_pos_min_imgs / float(M.size), num_neg_min_imgs / float(M.size)



def create_location_minimal_image_maps(image_id, top5map, top5map_small, loose, k=1):

    l_top5 = top5map[image_id, :, :]
    s_top5 = top5map_small[image_id, :, :]

    r, c = l_top5.shape
    M = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            self = l_top5[i, j]

            window = s_top5[i:i + 3, j:j + 3]  # get all the possible shrinks for this crop
            if loose:
                if self:  # if the current crop is correctly classified...
                    if not np.all(window):  # if any cell in the window is incorrectly classified...
                        M[i, j] = 1.  # ...the current crop is a positive minimal image. Otherwise, it's not minimal.
                else:  # if the current crop is incorrectly classified...
                    if np.any(window):  # if any cell in the window is correctly classified...
                        M[i, j] = -1.  # ...the current crop is a negative minimal image. Otherwise, it's not minimal.
            else:  # we are looking for strict minimal image maps
                if self:  # if the current crop is correctly classified...
                    if not np.any(window):  # if all crops in the window are incorrectly classified...
                        M[i, j] = 1.  # ...the current crop is a positive minimal image. Otherwise, it's not minimal.
                else:  # if the current crop is incorrectly classified...
                    if np.all(window):  # if all the crops in the window are correctly classified...
                        M[i, j] = -1.  # ...the current crop is a negative minimal image. Otherwise, it's not minimal.

    # calculate map statistics
    num_pos_min_imgs = (M > 0.).sum()
    num_neg_min_imgs = (M < 0.).sum()

    return num_pos_min_imgs / float(M.size), num_neg_min_imgs / float(M.size)



TOTAL = 1000
results = - np.ones([5, 2, 5, TOTAL, 2])

for idx_metric, crop_metric in enumerate(experiments.crop_sizes):

    with open(opt.log_dir_base + opt.name + '/maps/top/' + str(crop_metric) + '/maps.pkl', 'rb') as f:
        top5map = pickle.load(f)

    with open(opt.log_dir_base + opt.name + '/maps/top/' + str(crop_metric) + '/maps_small.pkl', 'rb') as f:
        top5map_small = pickle.load(f)

    print(idx_metric)
    sys.stdout.flush()
    for idx_loose, loose in enumerate([False, True]):
        for idx_k, k in enumerate([3]):  # enumerate([3, 5, 7, 11, 17]):
            print(k)
            sys.stdout.flush()
            for image_id in range(TOTAL):
                a, b = \
                    create_location_minimal_image_maps(image_id, top5map, top5map_small, loose, k)
                results[idx_metric][idx_loose][idx_k][image_id][0] = a
                results[idx_metric][idx_loose][idx_k][image_id][1] = b

        np.save(opt.log_dir_base + opt.name + '/tmp_results_' + opt.name + '_small.npy', results)



results = - np.ones([5, 2, 5, TOTAL, 2])

for idx_metric, crop_metric in enumerate(experiments.crop_sizes):

    with open(opt.log_dir_base + opt.name + '/maps/top_multi/' + str(crop_metric) + '/maps.pkl', 'rb') as f:
        top5map = pickle.load(f)

    with open(opt.log_dir_base + opt.name + '/maps/top/' + str(crop_metric) + '/maps_small.pkl', 'rb') as f:
        top5map_small = pickle.load(f)

    print(idx_metric)
    sys.stdout.flush()
    for idx_loose, loose in enumerate([False, True]):
        for idx_k, k in enumerate([3]):  # enumerate([3, 5, 7, 11, 17]):
            print(k)
            sys.stdout.flush()
            for image_id in range(TOTAL):
                a, b = \
                    create_location_minimal_image_maps_multi(image_id, top5map, top5map_small, loose, k)
                results[idx_metric][idx_loose][idx_k][image_id][0] = a
                results[idx_metric][idx_loose][idx_k][image_id][1] = b

        np.save(opt.log_dir_base + opt.name + '/tmp_results_multi' +  opt.name + '_small.npy', results)

print(':)')
sys.stdout.flush()
