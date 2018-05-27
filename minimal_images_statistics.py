import numpy as np
import sys
import experiments as experiments
import pickle

import os.path

ID = int(sys.argv[1:][0])

opt = experiments.opt[ID]


def create_location_minimal_image_maps_multi(image_id, top5map_general, loose, k=1):
    '''
    image_id (int): the small dataset id of the image we are finding minimal images for
    crop_metric (float): the crop metric we are referencing
    model_name (string): the model that we are referencing
    image_scale (float): the image scale we are referencing
    amount_loose (bool): amount_loose minimal images if True else strict minimal images
    k (int): the square size that we are looking for minimal image change within; should be even
    '''

    top5map_general = top5map_general[image_id, :, :]
    r, c = top5map_general.shape

    M = np.zeros((r, c))

    for ll in np.unique(top5map_general):

        top5map = np.copy(top5map_general)
        top5map[top5map_general == ll] = 1
        top5map[top5map_general != ll] = 0

        for i in range(r):
            for j in range(c):
                offset = int(k / 2)
                self = top5map[i, j]

                # make minimal image map
                if loose:
                    window = top5map[max(0, i - offset):min(r - 1, i + offset) + 1, max(0, j - offset):min(c - 1,
                                                                                                           j + offset) + 1]  # get the k-side-length window centered at current cell
                    if self:  # if the current cell is nonzero...
                        if not np.all(window):  # ...and if any part of the window is zero...
                            M[
                                i, j] = 1.  # ...this is a positive minimal image. If no other part of the window is zero, i.e. everything is nonzero, this is not a minimal image.
                    else:  # if the current cell is zero...
                        if np.any(window):  # ...and if any part of the window is nonzero...
                            M[
                                i, j] = -1.  # ...this is a negative minimal image. If no other part of the window is nonzero, i.e. everything is zero, this is not a minimal image.

                else:  # we are looking for strict minimal images
                    if self:  # if the current cell is nonzero...
                        top5map[i, j] = 0.  # temporarily set the current cell to zero
                        window = top5map[max(0, i - offset):min(r - 1, i + offset) + 1, max(0, j - offset):min(c - 1,
                                                                                                               j + offset) + 1]  # get the k-side-length window centered at current cell
                        if not np.any(window):  # ...and if no part of the window is nonzero...
                            M[
                                i, j] = 1.  # ...this is a positive minimal image. If some part of the window is nonzero, i.e. a surrounding pixel is nonzero, this is not a minimal image.
                        top5map[i, j] = self  # reset current cell
                    else:  # if the current cell is zero...
                        top5map[i, j] = 255.  # temporarily set the current cell to nonzero
                        window = top5map[max(0, i - offset):min(r - 1, i + offset) + 1, max(0, j - offset):min(c - 1,
                                                                                                               j + offset) + 1]  # get the k-side-length window centered at current cell
                        if np.all(window):  # ...and if the entire window is nonzero...
                            M[
                                i, j] = -1.  # ...this is a negative minimal image. If some part of the window is zero, i.e. a surrounding pixel is zero, this is not a minimal image.
                        top5map[i, j] = self  # reset current cell

    #  save map
    ''' 
    if amount_loose:
        np.save(PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale,
                                                     image_id) + '_lmap.npy', M)
    else:
        np.save(PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale,
                                                     image_id) + '_map.npy', M)
    '''

    # calculate map statistics
    num_pos_min_imgs = (M > 0.).sum()
    num_neg_min_imgs = (M < 0.).sum()

    return num_pos_min_imgs / float(M.size), num_neg_min_imgs / float(M.size)



def create_location_minimal_image_maps(image_id, top5map, loose, k=1):
    '''
    image_id (int): the small dataset id of the image we are finding minimal images for
    crop_metric (float): the crop metric we are referencing
    model_name (string): the model that we are referencing
    image_scale (float): the image scale we are referencing
    amount_loose (bool): amount_loose minimal images if True else strict minimal images
    k (int): the square size that we are looking for minimal image change within; should be even
    '''

    top5map = top5map[image_id, :, :]
    r, c = top5map.shape

    M = np.zeros((r, c))

    for i in range(r):
        for j in range(c):
            offset = int(k / 2)
            self = top5map[i, j]

            # make minimal image map
            if loose:
                window = top5map[max(0, i - offset):min(r - 1, i + offset) + 1, max(0, j - offset):min(c - 1,
                                                                                                       j + offset) + 1]  # get the k-side-length window centered at current cell
                if self:  # if the current cell is nonzero...
                    if not np.all(window):  # ...and if any part of the window is zero...
                        M[
                            i, j] = 1.  # ...this is a positive minimal image. If no other part of the window is zero, i.e. everything is nonzero, this is not a minimal image.
                else:  # if the current cell is zero...
                    if np.any(window):  # ...and if any part of the window is nonzero...
                        M[
                            i, j] = -1.  # ...this is a negative minimal image. If no other part of the window is nonzero, i.e. everything is zero, this is not a minimal image.

            else:  # we are looking for strict minimal images
                if self:  # if the current cell is nonzero...
                    top5map[i, j] = 0.  # temporarily set the current cell to zero
                    window = top5map[max(0, i - offset):min(r - 1, i + offset) + 1, max(0, j - offset):min(c - 1,
                                                                                                           j + offset) + 1]  # get the k-side-length window centered at current cell
                    if not np.any(window):  # ...and if no part of the window is nonzero...
                        M[
                            i, j] = 1.  # ...this is a positive minimal image. If some part of the window is nonzero, i.e. a surrounding pixel is nonzero, this is not a minimal image.
                    top5map[i, j] = self  # reset current cell
                else:  # if the current cell is zero...
                    top5map[i, j] = 255.  # temporarily set the current cell to nonzero
                    window = top5map[max(0, i - offset):min(r - 1, i + offset) + 1, max(0, j - offset):min(c - 1,
                                                                                                           j + offset) + 1]  # get the k-side-length window centered at current cell
                    if np.all(window):  # ...and if the entire window is nonzero...
                        M[
                            i, j] = -1.  # ...this is a negative minimal image. If some part of the window is zero, i.e. a surrounding pixel is zero, this is not a minimal image.
                    top5map[i, j] = self  # reset current cell

    #  save map
    ''' 
    if amount_loose:
        np.save(PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale,
                                                     image_id) + '_lmap.npy', M)
    else:
        np.save(PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale,
                                                     image_id) + '_map.npy', M)
    '''

    # calculate map statistics
    num_pos_min_imgs = (M > 0.).sum()
    num_neg_min_imgs = (M < 0.).sum()

    return num_pos_min_imgs / float(M.size), num_neg_min_imgs / float(M.size)

TOTAL = 1000
results = - np.ones([5, 2, 5, TOTAL, 2])


for idx_metric, crop_metric in enumerate(experiments.crop_sizes):

    with open(opt.log_dir_base + opt.name + '/maps/top/' + str(crop_metric) + '/maps.pkl', 'rb') as f:
        top5map = pickle.load(f)

    print(idx_metric)
    sys.stdout.flush()
    for idx_loose, loose in enumerate([False, True]):
        for idx_k, k in enumerate([3]):  # enumerate([3, 5, 7, 11, 17]):
            print(k)
            sys.stdout.flush()
            for image_id in range(TOTAL):
                a, b = \
                    create_location_minimal_image_maps(image_id, top5map, loose, k)
                results[idx_metric][idx_loose][idx_k][image_id][0] = a
                results[idx_metric][idx_loose][idx_k][image_id][1] = b

        np.save(opt.log_dir_base + opt.name + '/tmp_results_' + opt.name + '.npy', results)


results = - np.ones([5, 2, 5, TOTAL, 2])

for idx_metric, crop_metric in enumerate(experiments.crop_sizes):

    with open(opt.log_dir_base + opt.name + '/maps/top_multi/' + str(crop_metric) + '/maps.pkl', 'rb') as f:
        top5map = pickle.load(f)

    print(idx_metric)
    sys.stdout.flush()
    for idx_loose, loose in enumerate([False, True]):
        for idx_k, k in enumerate([3]):  # enumerate([3, 5, 7, 11, 17]):
            print(k)
            sys.stdout.flush()
            for image_id in range(TOTAL):
                a, b = \
                    create_location_minimal_image_maps_multi(image_id, top5map, loose, k)
                results[idx_metric][idx_loose][idx_k][image_id][0] = a
                results[idx_metric][idx_loose][idx_k][image_id][1] = b

        np.save(opt.log_dir_base + opt.name + '/tmp_results_multi' +  opt.name + '.npy', results)

print(':)')
sys.stdout.flush()
