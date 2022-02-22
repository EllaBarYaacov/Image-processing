# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged
from scipy import signal

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
import shutil
from imageio import imwrite
from scipy import ndimage

import sol4_utils


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """

    Ix = ndimage.filters.convolve(im, np.array([[1, 0, -1]]))
    Iy = ndimage.filters.convolve(im, np.array([[1, 0, -1]]).transpose())
    IxIy = sol4_utils.blur_spatial(Ix * Iy, 3)
    Ix2 = sol4_utils.blur_spatial(Ix ** 2, 3)
    Iy2 = sol4_utils.blur_spatial(Iy ** 2, 3)
    k = 0.04
    R = Ix2 * Iy2 - (IxIy ** 2) - k * (Ix2 + Iy2) ** 2
    yx_locations = np.where(non_maximum_suppression(R))
    return np.array([yx_locations[1], yx_locations[0]]).transpose()


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    result = []
    length = 2 * desc_rad + 1
    for corner_pos in pos:
        start_x = corner_pos[0] - desc_rad
        start_y = corner_pos[1] - desc_rad
        desc_x_indices = list(np.arange(start_x, start_x + length)) * length
        desc_y_indices = []
        for i in np.arange(start_y, start_y + length):
            desc_y_indices += [i] * length
        desc = ndimage.map_coordinates(im, [desc_y_indices, desc_x_indices], order=1, prefilter=False).reshape(
            (length, length))
        norm = np.linalg.norm(desc - np.mean(desc))
        normalized_desc = (desc - np.mean(desc)) / (norm if norm != 0 else 1)
        result.append(normalized_desc)
    return np.array(result)


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    corners = spread_out_corners(pyr[0], 7, 7, 3)
    descriptors = sample_descriptor(pyr[2], 0.25 * corners, 3)
    return corners, descriptors


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    mat1 = np.reshape(desc1, (desc1.shape[0], desc1.shape[1] * desc1.shape[2]))
    mat2 = np.reshape(desc2, (desc2.shape[0], desc2.shape[1] * desc2.shape[2]))
    score_mat = np.dot(mat1, mat2.transpose())
    sec_max_rows = np.partition(score_mat, -2)[:, -2: -1]
    sec_max_cols = np.partition(score_mat.transpose(), -2)[:, -2: -1]
    cond1 = score_mat >= sec_max_rows
    cond2 = (score_mat.transpose() >= sec_max_cols).transpose()
    cond3 = score_mat >= min_score
    return np.where(np.logical_and(np.logical_and(cond1, cond2), cond3))


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    homogenic_cords = np.concatenate((pos1, np.ones((pos1.shape[0], 1))), axis=1)
    mid_result = np.dot(H12, homogenic_cords.transpose())
    result = np.stack((mid_result[0] / mid_result[2], mid_result[1] / mid_result[2]))
    return result.transpose()


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    best_inliers_indexes = []
    for i in range(num_iter):
        if points1.shape[0] == 0:
            f = 9
        matches_indexes = np.random.randint(0, points1.shape[0], 2)
        transform_M = estimate_rigid_transform(points1[matches_indexes], points2[matches_indexes], translation_only)
        estimate_p2 = apply_homography(points1, transform_M)
        errors = np.sqrt(np.sum(((estimate_p2 - points2) ** 2), axis=1))
        inliers_true_false = errors < inlier_tol
        inliers_num = np.count_nonzero(inliers_true_false)
        if inliers_num > len(best_inliers_indexes):
            best_inliers_indexes = np.where(inliers_true_false)[0]
    best_homography = estimate_rigid_transform(points1[best_inliers_indexes], points2[best_inliers_indexes],
                                               translation_only)
    return best_homography, best_inliers_indexes


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """

    plt.imshow(np.hstack([im1, im2]), cmap='gray')
    plt.scatter(points1.transpose()[0], points1.transpose()[1], c='r', s=1)
    plt.scatter(points2.transpose()[0] + im1.shape[1], points2.transpose()[1], c='r', s=1)
    shifted_points2 = np.array([points2.transpose()[0] + im1.shape[1], points2.transpose()[1]]).transpose()
    outliers = list(set(np.arange(points1.shape[0])) - set(inliers))
    pairs = np.array([points1, shifted_points2])
    plt.plot(pairs[:, inliers, 0], pairs[:, inliers, 1], c='y', lw=.4, )
    plt.plot(pairs[:, outliers, 0], pairs[:, outliers, 1], c='b', lw=.4, )
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    left_H = right_H = np.eye(3)
    H2m = [np.eye(3)]
    for i in np.arange(m - 1, -1, -1):
        new_H = np.dot(left_H, H_succesive[i])
        new_H /= new_H[2, 2]
        H2m.insert(0, new_H)
        left_H = new_H
    for i in np.arange(m, len(H_succesive)):
        new_H = np.dot(right_H, np.linalg.inv(H_succesive[i]))
        new_H /= new_H[2, 2]
        H2m.append(new_H)
        right_H = new_H
    return H2m


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    new_corners = apply_homography(np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]]), homography)
    min_x = np.min(new_corners.transpose()[0])
    max_x = np.max(new_corners.transpose()[0])
    min_y = np.min(new_corners.transpose()[1])
    max_y = np.max(new_corners.transpose()[1])
    return np.array([[min_x, min_y], [max_x, max_y]]).astype(np.int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    top_left, bottom_right = compute_bounding_box(homography, image.shape[1], image.shape[0])
    transformed_coords = np.array(np.meshgrid(np.arange(top_left[0], bottom_right[0] + 1), np.arange(top_left[1], bottom_right[1] + 1)))
    transformed_coords_flat = np.array([transformed_coords[0].ravel(), transformed_coords[1].ravel()]).transpose()
    original_coords = apply_homography(transformed_coords_flat, np.linalg.inv(homography))
    interpulated_values = ndimage.map_coordinates(image,[original_coords.transpose()[1], original_coords.transpose()[0]],order=1, prefilter=False)
    return interpulated_values.reshape(transformed_coords[0].shape)


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()


if __name__ == '__main__':
    # p1 = np.random.randint(1, 10, (10, 2))
    # p2 = np.random.randint(1, 10, (10, 2))
    # h, i = ransac_homography(p1, p2, 10, 4, translation_only=False)
    # print(h)
    # print(i)

    # image_path = '/Users/ellaby/Documents/year D/image_processing/exes/ex2-ellaby/external/monkey.jpg'
    # im1 = sol4_utils.read_image(image_path, 1)
    #
    # t = np.array([[1, 0, 3], [0, 2, 0], [0, 0, 1]])
    # i = warp_channel(im1, t)
    # plt.imshow(i)
    # plt.show()
    # im2 = sol4_utils.read_image(image_path, 1)
    # points1 = np.random.randint(0, 400, (100, 2))
    # points2 = np.random.randint(0, 400, (100, 2))
    # display_matches(im1, im2, points1, points2, np.arange(50))

    # a = [1,2,3,4]
    # print(a)
    # a.insert(0, 18)
    # print(a)
    # a.append(40)
    # print(a)
    # a = np.array([[1, 2], [3, 4]])
    # b = np.array([[4, 6], [7, 8]])
    # print(np.dot(a,b))
    # print(np.dot(b, a))
    # c = np.meshgrid(np.arange(3,7),np.arange(11, 18))
    # top_left = [3,7]
    # bottom_right = [8, 15]
    #
    # transformed_coords = np.meshgrid(np.arange(top_left[0], bottom_right[0] + 1), np.arange(top_left[1],
    #                                                                                         bottom_right[1] + 1))
    # transformed_coords = np.array(transformed_coords)
    # transformed_coords_flat = np.array([transformed_coords[0].ravel(), transformed_coords[1].ravel()]).transpose()
    # print(transformed_coords)
    # print(transformed_coords_flat)
    print(np.array(np.meshgrid([1, 2, 3], [4, 5, 6, 7, 8, 9])))
