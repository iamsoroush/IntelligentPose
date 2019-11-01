from configobj import ConfigObj
from time import time

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfb
from tensorflow import keras as tfk
import cv2
tfkl = tfk.layers
tfkb = tfk.backend


class CPM:
    def __init__(self, input_shape=(None, None, 3), dropout_rate=0.1, n_parts=16):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.n_parts = n_parts

    def create_model(self):
        input_image = tfk.Input(shape=self.input_shape)

        features = self._feature_extractor(input_image)
        first_stage_believes = self._cpm_first_stage(input_image)

        second_stage_believes = self._cpm_second_stage(features, first_stage_believes, prefix='stage2_')

        third_stage_believes = self._cpm_second_stage(features, second_stage_believes, prefix='stage3_')

        fourth_stage_believes = self._cpm_second_stage(features, third_stage_believes, prefix='stage4_')

        out = tfkl.Activation('sigmoid', name='final_heatmaps')(fourth_stage_believes)

        model = tfk.Model(input_image, out)
        return model

    def _cpm_first_stage(self, input_image):
        y = self._conv2d(input_image, filters=16, kernel_size=(3, 3))
        y = self._conv2d(y, filters=16, kernel_size=(3, 3))
        x = self._add_skip_connection(input_image, y)
        x = tfkl.MaxPooling2D(2, strides=2, padding='same')(x)

        y = self._conv2d(x, filters=32, kernel_size=(3, 3))
        y = self._conv2d(y, filters=32, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)
        x = tfkl.MaxPooling2D(2, strides=2, padding='same')(x)

        y = self._conv2d(x, filters=64, kernel_size=(3, 3))
        y = self._conv2d(y, filters=64, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)
        x = tfkl.MaxPooling2D(2, strides=2, padding='same')(x)

        y = self._conv2d(x, filters=128, kernel_size=(2, 2))
        y = self._conv2d(y, filters=128, kernel_size=(2, 2))
        x = self._add_skip_connection(x, y)

        y = self._conv2d(x, filters=256, kernel_size=(3, 3))
        y = self._conv2d(y, filters=256, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)

        x = self._conv2d(x, 64, kernel_size=(1, 1))

        out = tfkl.Conv2D(self.n_parts, (1, 1), padding='same', activation=None, name='stage1_repr')(x)
        out = tfkl.BatchNormalization()(out)
        out = tfkl.Activation('relu')(out)
        out = tfkl.SpatialDropout2D(self.dropout_rate)(out)
        return out

    def _cpm_second_stage(self, extracted_features, former_believes, prefix):
        input_tensor = tfkl.concatenate([extracted_features, former_believes],
                                        axis=-1, name=prefix + 'concat')

        y = self._conv2d(input_tensor, filters=64, kernel_size=(4, 4))
        y = self._conv2d(y, filters=64, kernel_size=(4, 4))
        x = self._add_skip_connection(input_tensor, y)

        y = self._conv2d(x, filters=128, kernel_size=(4, 4))
        y = self._conv2d(y, filters=128, kernel_size=(4, 4))
        x = self._add_skip_connection(x, y)

        y = self._conv2d(x, filters=256, kernel_size=(3, 3))
        y = self._conv2d(y, filters=256, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)

        x = self._conv2d(x, 64, kernel_size=(1, 1))
        out = tfkl.Conv2D(self.n_parts, (1, 1), padding='same', name=prefix + 'repr')(x)
        return out

    def _feature_extractor(self, input_image):
        y = self._conv2d(input_image, filters=16, kernel_size=(3, 3))
        y = self._conv2d(y, filters=16, kernel_size=(3, 3))
        x = self._add_skip_connection(input_image, y)
        x = tfkl.MaxPooling2D(2, strides=2, padding='same')(x)

        y = self._conv2d(x, filters=32, kernel_size=(3, 3))
        y = self._conv2d(y, filters=32, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)
        x = tfkl.MaxPooling2D(2, strides=2, padding='same')(x)

        y = self._conv2d(x, filters=64, kernel_size=(3, 3))
        y = self._conv2d(y, filters=64, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)
        x = tfkl.MaxPooling2D(2, strides=2, padding='same')(x)

        y = self._conv2d(x, filters=128, kernel_size=(2, 2))
        y = self._conv2d(y, filters=128, kernel_size=(2, 2))
        x = self._add_skip_connection(x, y)

        x = tfkl.Conv2D(self.n_parts, (1, 1), padding='same')(x)
        return x

    def _conv2d(self, x, filters, kernel_size):
        out = tfkl.Conv2D(filters, kernel_size, padding='same')(x)
        out = tfkl.BatchNormalization()(out)
        out = tfkl.Activation('relu')(out)
        out = tfkl.SpatialDropout2D(self.dropout_rate)(out)
        return out

    def _add_skip_connection(self, x, y, scale_factor=0.5):
        channels = tfkb.int_shape(y)[-1]
        shortcut_branch = tfkl.Conv2D(filters=channels, kernel_size=(1, 1), padding='same')(x)
        out = self._weighted_add(shortcut_branch, y, scale_factor)
        return tfkl.Activation('relu')(out)

    @staticmethod
    def _weighted_add(shortcut_branch, inception_branch, scale_factor):
        return tfkl.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                           arguments={'scale': scale_factor})([shortcut_branch, inception_branch])


class StackedHourglassNetwork:
    pass


class OpenPose:
    map_idx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
               [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
               [55, 56], [37, 38], [45, 46]]
    # find connection in the specified sequence, center 29 is in the position 15
    limb_seq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
              [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
              [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    def __init__(self, weights_path, config_path, n_scales=1):
        self.weights_path = weights_path
        self.config_path = config_path
        self.params, self.model_params = self._read_config()
        self.n_scales = n_scales
        open_pose_obj = OpenPoseModel()
        self.model = open_pose_obj.create_model()
        self._load_model()
        self.fe = FeatureExtractor()
        self.n_joints = 18

    def _read_config(self):
        config = ConfigObj(self.config_path)
        param = config['param']
        model_id = param['modelID']
        model = config['models'][model_id]
        model['boxsize'] = int(model['boxsize'])
        model['stride'] = int(model['stride'])
        model['padValue'] = int(model['padValue'])
        param['octave'] = int(param['octave'])
        param['use_gpu'] = int(param['use_gpu'])
        param['starting_range'] = float(param['starting_range'])
        param['ending_range'] = float(param['ending_range'])
        param['scale_search'] = list(map(float, param['scale_search']))
        param['thre1'] = float(param['thre1'])
        param['thre2'] = float(param['thre2'])
        param['thre3'] = float(param['thre3'])
        param['mid_num'] = int(param['mid_num'])
        param['min_num'] = int(param['min_num'])
        param['crop_ratio'] = float(param['crop_ratio'])
        param['bbox_ratio'] = float(param['bbox_ratio'])
        param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])
        return param, model

    def draw_pose(self, img, inference_h, inference_w, n_scales=None):
        if n_scales is not None:
            org_n_scales = self.n_scales
            self.n_scales = n_scales
        org_h, org_w, _ = img.shape
        resized = tf.image.resize_with_pad(img, inference_h, inference_w).numpy()
        peaks, subset, candidate = self._complete_inference(resized, n_scales)

        if not subset.any():
            drawed = None
        else:
            transformed_candidate = self.inverse_transform_kps(org_h, org_w, inference_h, inference_w, candidate)
            drawed = self.draw_inverse_transformed_parts(img, peaks, subset, transformed_candidate)
        if n_scales is not  None:
            self.n_scales = org_n_scales
        return drawed

    def get_img_pose_kps(self, img, inference_h, inference_w, n_scales):
        """Returns computed features for first pose found in the image."""

        org_h, org_w, _ = img.shape
        resized = tf.image.resize_with_pad(img, inference_h, inference_w).numpy()
        all_peaks, subset, candidate = self._complete_inference(resized, n_scales)

        if not subset.any():
            kps = None
        else:
            person = subset[0]
            kps = self._extract_keypoints(person, candidate)

        return kps

    @staticmethod
    def _extract_keypoints(person_subset, candidate_arr):
        kps = list()
        for i in range(18):
            kp_ind = person_subset[i].astype(np.int)
            if kp_ind == -1:
                kps.append(None)
            else:
                kps.append(candidate_arr[kp_ind, 0: 2].astype(np.int))
        return kps

    def _complete_inference(self, img, n_scales):
        if n_scales is not None:
            org_n_scales = self.n_scales
            self.n_scales = n_scales
        heatmap_avg, paf_avg = self._get_hm_paf_av(img)
        all_peaks = self._get_peaks(heatmap_avg, self.params['thre1'])
        connection_all, special_k = self._get_connections(paf_avg, all_peaks, self.params['thre2'], img.shape)
        subset, candidate = self._get_subset(all_peaks, special_k, connection_all)
        if n_scales is not None:
            self.n_scales = org_n_scales
        return all_peaks, subset, candidate

    def predict(self, img):
        """Returns keypoints, subset and candidate.

        img should be of type BGR.
        """

        t = time()
        heatmap_avg, paf_avg = self._get_hm_paf_av(img)
        print('s1: ', time() - t)
        t1 = time()
        all_peaks = self._get_peaks(heatmap_avg, self.params['thre1'])
        print('s2: ', time() - t1)
        t1 = time()
        connection_all, special_k = self._get_connections(paf_avg, all_peaks, self.params['thre2'], img.shape)
        print('s3: ', time() - t1)
        t1 = time()
        subset, candidate = self._get_subset(all_peaks, special_k, connection_all)
        print('s4: ', time() - t1)
        print('Execution time: ', time() - t)
        return all_peaks, subset, candidate

    def compare_draw(self, img, target_kps, inference_h, inference_w, n_scales=1, th=5):
        correct_color = OpenPose.colors[9]
        wrong_color = [255, 255, 0]
        target_features = self.fe.generate_features(target_kps)

        org_h, org_w, _ = img.shape
        max_radius = org_h // 5

        resized = tf.image.resize_with_pad(img, inference_h, inference_w).numpy()
        t = time()
        peaks, subset, candidate = self._complete_inference(resized, n_scales)
        print('complete inference: ', time() - t)

        if not subset.any():
            return img
        else:
            transformed_candidate = self.inverse_transform_kps(org_h, org_w, inference_h, inference_w, candidate)

        person = subset[0]
        t = time()
        kps = self._extract_keypoints(person, transformed_candidate)
        features = self.fe.generate_features(kps)
        print('kp extraction and feature generation: ', time() - t)

        for kp in kps:
            if kp is not None:
                cv2.circle(img, (int(kp[0]), int(kp[1])), 4, correct_color, thickness=-1)

        t = time()
        for i in range(len(features)):
            f = features[i]
            ft = target_features[i]
            if (f is not None) and (ft is not None):
                f_diff = np.abs(f - ft)
                if f_diff >= th:
                    radius = int(max_radius * f_diff / 360)
                    failure_overlay = img.copy()
                    kp = kps[self.fe.points_comb[i][1]]
                    cv2.circle(failure_overlay, (int(kp[0]), int(kp[1])), radius, wrong_color, thickness=-1)
                    img = cv2.addWeighted(img, 0.4, failure_overlay, 0.6, 0)
        print('difference drawing: ', time() - t)
        stick_width = 4

        for i in range(17):
            index = person[np.array(OpenPose.limb_seq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = img.copy()
            y = transformed_candidate[index.astype(int), 0]
            x = transformed_candidate[index.astype(int), 1]
            kps.append([])
            m_x = np.mean(x)
            m_y = np.mean(y)
            length = np.sqrt(np.power(x[0] - x[1], 2) + np.power(y[0] - y[1], 2))
            angle = np.degrees(np.arctan2(x[0] - x[1], y[0] - y[1]))
            polygon = cv2.ellipse2Poly((int(m_y), int(m_x)),
                                       (int(length / 2), stick_width),
                                       int(angle),
                                       0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, OpenPose.colors[i])
            img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)
        return img

    @staticmethod
    def inverse_transform_kps(org_h, org_w, h, w, candidate):
        kps = candidate[:, 0: 2].astype(np.int)
        scale_factor = np.max([org_h, org_w]) / h
        transformed_candidate = np.zeros((candidate.shape[0], 3))
        if org_h > org_w:
            resized_w = org_w / scale_factor
            border = (w - resized_w) / 2
            for i, kp in enumerate(kps):
                transformed_candidate[i, 0] = scale_factor * (kp[0] - border)
                transformed_candidate[i, 1] = scale_factor * kp[1]
                transformed_candidate[i, 2] = candidate[i, 2]
        else:
            resized_h = org_h / scale_factor
            border = (h - resized_h) / 2
            for i, kp in enumerate(kps):
                transformed_candidate[i, 0] = scale_factor * kp[0]
                transformed_candidate[i, 1] = scale_factor * (kp[1] - border)
                transformed_candidate[i, 2] = candidate[i, 2]
        return transformed_candidate

    @staticmethod
    def draw_inverse_transformed_parts(img, peaks, subset, transformed_candidate):
        valid_indices = subset.flatten().astype(np.int).tolist()[: 18]
        for i in range(18):
            for j in range(len(peaks[i])):
                ind = peaks[i][j][-1]
                if ind in valid_indices:
                    c = transformed_candidate[ind]
                    cv2.circle(img, (int(c[0]), int(c[1])), 4, OpenPose.colors[i], thickness=-1)

        stick_width = 4

        for i in range(17):
            for n in range(len(subset)):
                index = subset[n][np.array(OpenPose.limb_seq[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = img.copy()
                y = transformed_candidate[index.astype(int), 0]
                x = transformed_candidate[index.astype(int), 1]
                m_x = np.mean(x)
                m_y = np.mean(y)
                length = np.sqrt(np.power(x[0] - x[1], 2) + np.power(y[0] - y[1], 2))
                angle = np.degrees(np.arctan2(x[0] - x[1], y[0] - y[1]))
                polygon = cv2.ellipse2Poly((int(m_y), int(m_x)),
                                           (int(length / 2), stick_width),
                                           int(angle),
                                           0,
                                           360,
                                           1)
                cv2.fillConvexPoly(cur_canvas, polygon, OpenPose.colors[i])
                img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)
        return img

    @staticmethod
    def draw_parts(canvas, peaks, subset, candidate):
        valid_indices = subset.flatten().astype(np.int).tolist()
        for i in range(18):
            for j in range(len(peaks[i])):
                peak = peaks[i][j]
                if int(peak[-1]) not in valid_indices:
                    continue
                cv2.circle(canvas, peak[0:2], 4, OpenPose.colors[i], thickness=-1)

        stick_width = 4

        for i in range(17):
            for n in range(len(subset)):
                index = subset[n][np.array(OpenPose.limb_seq[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                y = candidate[index.astype(int), 0]
                x = candidate[index.astype(int), 1]
                m_x = np.mean(x)
                m_y = np.mean(y)
                length = np.sqrt(np.power(x[0] - x[1], 2) + np.power(y[0] - y[1], 2))
                angle = np.degrees(np.arctan2(x[0] - x[1], y[0] - y[1]))
                polygon = cv2.ellipse2Poly((int(m_y), int(m_x)),
                                           (int(length / 2), stick_width),
                                           int(angle),
                                           0,
                                           360,
                                           1)
                cv2.fillConvexPoly(cur_canvas, polygon, OpenPose.colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        return canvas

    @staticmethod
    def _get_peaks(heatmap_avg, thre1):
        all_peaks = []
        peak_counter = 0
        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            # _map = gaussian_filter(map_ori, sigma=3)
            _map = map_ori
            map_left = np.zeros(_map.shape)
            map_left[1:, :] = _map[:-1, :]
            map_right = np.zeros(_map.shape)
            map_right[:-1, :] = _map[1:, :]
            map_up = np.zeros(_map.shape)
            map_up[:, 1:] = _map[:, :-1]
            map_down = np.zeros(_map.shape)
            map_down[:, :-1] = _map[:, 1:]

            peaks_binary = np.logical_and.reduce((_map >= map_left,
                                                  _map >= map_right,
                                                  _map >= map_up,
                                                  _map >= map_down,
                                                  _map > thre1))
            nz = np.nonzero(peaks_binary)
            peaks = list(zip(nz[1], nz[0]))  # note reverse
            n_peaks = len(peaks)
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peaks_with_score_and_id = [peaks_with_score[i - peak_counter] + (i,) for i in range(peak_counter,
                                                                                                peak_counter + n_peaks)]
            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)
        return all_peaks

    @staticmethod
    def _get_connections(paf_avg, all_peaks, thre2, img_shape):
        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(OpenPose.map_idx)):
            score_mid = paf_avg[:, :, [x - 19 for x in OpenPose.map_idx[k]]]
            cand_a = all_peaks[OpenPose.limb_seq[k][0] - 1]
            cand_b = all_peaks[OpenPose.limb_seq[k][1] - 1]
            n_a = len(cand_a)
            n_b = len(cand_b)
            if n_a != 0 and n_b != 0:
                connection_candidate = []
                for i in range(n_a):
                    for j in range(n_b):
                        vec = np.subtract(cand_b[j][:2], cand_a[i][:2])
                        norm = np.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        # failure case when 2 body parts overlaps
                        if norm == 0:
                            continue
                        vec = np.divide(vec, norm)

                        start_end = list(zip(np.linspace(cand_a[i][0], cand_b[j][0], num=mid_num),
                                             np.linspace(cand_a[i][1], cand_b[j][1], num=mid_num)))

                        vec_x = np.array(
                            [score_mid[int(round(start_end[I][1])), int(round(start_end[I][0])), 0]
                             for I in range(len(start_end))])
                        vec_y = np.array(
                            [score_mid[int(round(start_end[I][1])), int(round(start_end[I][0])), 1]
                             for I in range(len(start_end))])

                        score_mid_pts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_mid_pts) / len(score_mid_pts) + min(
                            0.5 * img_shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_mid_pts > thre2)[0]) > 0.8 * len(
                            score_mid_pts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior,
                                                         score_with_dist_prior + cand_a[i][2] + cand_b[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [cand_a[i][3], cand_b[j][3], s, i, j]])
                        if len(connection) >= min(n_a, n_b):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])
        return connection_all, special_k

    @staticmethod
    def _get_subset(all_peaks, special_k, connection_all):
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(OpenPose.map_idx)):
            if k not in special_k:
                part_as = connection_all[k][:, 0]
                part_bs = connection_all[k][:, 1]
                index_a, index_b = np.array(OpenPose.limb_seq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][index_a] == part_as[i] or subset[j][index_b] == part_bs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][index_b] != part_bs[i]:
                            subset[j][index_b] = part_bs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][index_b] = part_bs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[index_a] = part_as[i]
                        row[index_b] = part_bs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # delete some rows of subset which has few parts occur
        delete_idx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                delete_idx.append(i)
        subset = np.delete(subset, delete_idx, axis=0)
        return subset, candidate

    def _get_hm_paf_av(self, img):
        """Returns heatmaps and pafs, (ims_size, 19) and (img_size, 38)"""
        multiplier = [i * self.model_params['boxsize'] / img.shape[0] for i in self.params['scale_search']]

        if self.n_scales == 1:
            scale = multiplier[0]
            output_blobs, padded_resized_img, pad = self._infere(img, scale)

            # extract outputs, resize, and remove padding
            heatmap = self._get_heatmap(output_blobs,
                                        self.model_params['stride'],
                                        padded_resized_img.shape,
                                        img.shape,
                                        pad)
            paf = self._get_paf(output_blobs,
                                self.model_params['stride'],
                                padded_resized_img.shape,
                                img.shape,
                                pad)
            return heatmap, paf

        heatmap_avg = np.zeros((img.shape[0], img.shape[1], 19))
        paf_avg = np.zeros((img.shape[0], img.shape[1], 38))

        for m in range(len(multiplier)):
            if m >= self.n_scales:
                break
            scale = multiplier[m]

            output_blobs, padded_resized_img, pad = self._infere(img, scale)

            # extract outputs, resize, and remove padding
            heatmap = self._get_heatmap(output_blobs,
                                        self.model_params['stride'],
                                        padded_resized_img.shape,
                                        img.shape,
                                        pad)
            paf = self._get_paf(output_blobs,
                                self.model_params['stride'],
                                padded_resized_img.shape,
                                img.shape,
                                pad)
            heatmap_avg = heatmap_avg + heatmap / self.n_scales
            paf_avg = paf_avg + paf / self.n_scales
        return heatmap_avg, paf_avg

    def _infere(self, img, scale):
        stride = self.model_params['stride']
        pad_value = self.model_params['padValue']
        resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        padded_resized_img, pad = self._pad_right_down_corner(resized_img, stride, pad_value)

        input_img = padded_resized_img[np.newaxis, :, :, :]
        t = time()
        output_blobs = self.model.predict(input_img)
        print('inference: ', time() - t)
        return output_blobs, padded_resized_img, pad

    @staticmethod
    def _get_heatmap(output_blobs, stride, padded_resized_shape, img_shape, pad):
        heatmap = np.squeeze(output_blobs[1])
        heatmap = cv2.resize(heatmap,
                             (0, 0),
                             fx=stride,
                             fy=stride,
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:padded_resized_shape[0] - pad[2],
                          :padded_resized_shape[1] - pad[3],
                          :]
        heatmap = cv2.resize(heatmap,
                             (img_shape[1], img_shape[0]),
                             interpolation=cv2.INTER_CUBIC)
        return heatmap

    @staticmethod
    def _get_paf(output_blobs, stride, padded_resized_shape, img_shape, pad):
        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf,
                         (0, 0),
                         fx=stride,
                         fy=stride,
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:padded_resized_shape[0] - pad[2],
                  :padded_resized_shape[1] - pad[3],
                  :]
        paf = cv2.resize(paf, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)
        return paf

    @staticmethod
    def _pad_right_down_corner(img, stride, pad_value):
        h = img.shape[0]
        w = img.shape[1]

        pad = 4 * [None]
        pad[0] = 0  # up
        pad[1] = 0  # left
        pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
        pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

        img_padded = img
        pad_up = np.tile(img_padded[0:1, :, :] * 0 + pad_value, (pad[0], 1, 1))
        img_padded = np.concatenate((pad_up, img_padded), axis=0)
        pad_left = np.tile(img_padded[:, 0:1, :] * 0 + pad_value, (1, pad[1], 1))
        img_padded = np.concatenate((pad_left, img_padded), axis=1)
        pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + pad_value, (pad[2], 1, 1))
        img_padded = np.concatenate((img_padded, pad_down), axis=0)
        pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + pad_value, (1, pad[3], 1))
        img_padded = np.concatenate((img_padded, pad_right), axis=1)

        return img_padded, pad

    def _load_model(self):
        self.model.load_weights(self.weights_path)
        print('Model loaded successfully')


class OpenPoseModel:

    def __init__(self):
        pass

    def create_model(self):
        input_shape = (None, None, 3)
        stages = 6
        np_branch1 = 38
        np_branch2 = 19

        input_tensor = tfkl.Input(input_shape)  # Input must be RGB and (0, 255
        normalized_input = tfkl.Lambda(lambda x: x / 256 - 0.5)(input_tensor)  # [-0.5, 0.5]

        # VGG
        stage0_out = self._vgg_block(normalized_input)

        # stage 1
        stage1_branch1_out = self._stage1_block(stage0_out, np_branch1, 1)
        stage1_branch2_out = self._stage1_block(stage0_out, np_branch2, 2)
        x = tfkl.Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

        # stage t >= 2
        for sn in range(2, stages + 1):
            stage_t_branch1_out = self._stage_t_block(x, np_branch1, sn, 1)
            stage_t_branch2_out = self._stage_t_block(x, np_branch2, sn, 2)
            if sn < stages:
                x = tfkl.Concatenate()([stage_t_branch1_out, stage_t_branch2_out, stage0_out])

        model = tfk.Model(input_tensor, [stage_t_branch1_out, stage_t_branch2_out])
        return model

    def _vgg_block(self, x):
        # Block 1
        x = self._conv(x, 64, 3, "conv1_1")
        x = self._relu(x)
        x = self._conv(x, 64, 3, "conv1_2")
        x = self._relu(x)
        x = self._pooling(x, 2, 2, "pool1_1")

        # Block 2
        x = self._conv(x, 128, 3, "conv2_1")
        x = self._relu(x)
        x = self._conv(x, 128, 3, "conv2_2")
        x = self._relu(x)
        x = self._pooling(x, 2, 2, "pool2_1")

        # Block 3
        x = self._conv(x, 256, 3, "conv3_1")
        x = self._relu(x)
        x = self._conv(x, 256, 3, "conv3_2")
        x = self._relu(x)
        x = self._conv(x, 256, 3, "conv3_3")
        x = self._relu(x)
        x = self._conv(x, 256, 3, "conv3_4")
        x = self._relu(x)
        x = self._pooling(x, 2, 2, "pool3_1")

        # Block 4
        x = self._conv(x, 512, 3, "conv4_1")
        x = self._relu(x)
        x = self._conv(x, 512, 3, "conv4_2")
        x = self._relu(x)

        # Additional non vgg layers
        x = self._conv(x, 256, 3, "conv4_3_CPM")
        x = self._relu(x)
        x = self._conv(x, 128, 3, "conv4_4_CPM")
        x = self._relu(x)
        return x

    def _stage1_block(self, x, num_p, branch):
        x = self._conv(x, 128, 3, "conv5_1_CPM_L%d" % branch)
        x = self._relu(x)
        x = self._conv(x, 128, 3, "conv5_2_CPM_L%d" % branch)
        x = self._relu(x)
        x = self._conv(x, 128, 3, "conv5_3_CPM_L%d" % branch)
        x = self._relu(x)
        x = self._conv(x, 512, 1, "conv5_4_CPM_L%d" % branch)
        x = self._relu(x)
        x = self._conv(x, num_p, 1, "conv5_5_CPM_L%d" % branch)
        return x

    def _stage_t_block(self, x, num_p, stage, branch):
        x = self._conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch))
        x = self._relu(x)
        x = self._conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch))
        x = self._relu(x)
        x = self._conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch))
        x = self._relu(x)
        x = self._conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch))
        x = self._relu(x)
        x = self._conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch))
        x = self._relu(x)
        x = self._conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch))
        x = self._relu(x)
        x = self._conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch))
        return x

    @staticmethod
    def _conv(x, nf, ks, name):
        out = tfkl.Conv2D(nf, (ks, ks), padding='same', name=name)(x)
        return out

    @staticmethod
    def _relu(x):
        return tfkl.Activation('relu')(x)

    @staticmethod
    def _pooling(x, ks, st, name):
        x = tfkl.MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
        return x


class FastOpenPose:
    map_idx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
               [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
               [55, 56], [37, 38], [45, 46]]
    # find connection in the specified sequence, center 29 is in the position 15
    limb_seq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
              [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
              [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    def __init__(self,
                 weights_path,
                 config_path,
                 input_shape=(184, 184),
                 gaussian_filtering=True):
        self.openpose_model = FastOpenPoseModel(weights_path,
                                                config_path,
                                                input_shape,
                                                gaussian_filtering)
        self.model = self.openpose_model.load_model()
        self.fe = FeatureExtractor()
        self.n_joints = 18
        self.n_limbs = 17

    def compare_draw(self, img, target_kps, th=5):
        correct_color = OpenPose.colors[9]
        wrong_color = [0, 0, 255]
        target_features = self.fe.generate_features(target_kps)

        org_h, org_w, _ = img.shape

        resized = tf.image.resize_with_pad(img, self.openpose_model.input_h, self.openpose_model.input_w).numpy()
        t = time()
        peaks, subset, candidate = self._inference(resized)
        print('complete inference: ', time() - t)

        if not subset.any():
            return img
        else:
            transformed_candidate = self.inverse_transform_kps(org_h, org_w,
                                                               self.openpose_model.input_h,
                                                               self.openpose_model.input_w,
                                                               candidate)

        drawed = img.copy()

        for person in subset:
            t = time()
            kps = self._extract_keypoints(person, transformed_candidate)
            print('KP extraction: ', time() - t)
            t = time()
            features = self.fe.generate_features(kps)
            print('Feature generation: ', time() - t)

            self._draw_kps(drawed, kps, correct_color)

            t = time()
            drawed = self._draw_errors(drawed, features, target_features, kps, th, wrong_color)
            print('Error drawing: ', time() - t)

            t = time()
            drawed = self._draw_connections(drawed, person, transformed_candidate)
            print('Connections drawing: ', time() - t)
        return drawed

    @staticmethod
    def _draw_kps(img, kps, color):
        for kp in kps:
            if kp is not None:
                cv2.circle(img, (int(kp[0]), int(kp[1])), 4, color, thickness=-1)

    def _draw_errors(self, img, features, target_features, kps, threshold, color):
        x_min, y_min, x_max, y_max = self._get_ul_lr(kps)
        diag = np.sqrt(np.power(x_max - x_min, 2) + np.power(y_max - y_min, 2)).astype(np.int)
        max_radius = diag // 4
        errors = list()
        failure_overlay = img.copy()
        for i in range(len(features)):
            f = features[i]
            ft = target_features[i]
            if (f is not None) and (ft is not None):
                f_diff = np.abs(f - ft)
                if f_diff >= threshold:
                    errors.append(f_diff / 360)
                    radius = int(max_radius * f_diff / 360)
                    kp = kps[self.fe.points_comb[i][1]]
                    # cv2.putText(failure_overlay,
                    #             str(int(f)),
                    #             (int(kp[0]), int(kp[1])),
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             1,
                    #             (255, 255, 255),
                    #             2)

                    cv2.circle(failure_overlay, (int(kp[0]), int(kp[1])), radius, color, thickness=-1)
        # error = None
        # if len(errors) > 0:
        #     error = int(np.sum(errors) * 100)
        # cv2.putText(failure_overlay,
        #             'error: {}'.format(error),
        #             (x_min, max(y_min - diag // 10, 10)),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             1,
        #             (255, 255, 255),
        #             2)
        if len(errors) == 0:
            cv2.putText(failure_overlay,
                        "That's OK!",
                        (x_min, max(y_min - diag // 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)
        else:
            cv2.putText(failure_overlay,
                        'Do it better!',
                        (x_min, max(y_min - diag // 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2)

        drawed = cv2.addWeighted(img, 0.4, failure_overlay, 0.6, 0)
        return drawed

    def _draw_connections(self, img, person, transformed_candidate):

        stick_width = 4

        for i in range(self.n_limbs):
            index = person[np.array(OpenPose.limb_seq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = img.copy()
            y = transformed_candidate[index.astype(int), 0]
            x = transformed_candidate[index.astype(int), 1]
            m_x = np.mean(x)
            m_y = np.mean(y)
            length = np.sqrt(np.power(x[0] - x[1], 2) + np.power(y[0] - y[1], 2))
            angle = np.degrees(np.arctan2(x[0] - x[1], y[0] - y[1]))
            polygon = cv2.ellipse2Poly((int(m_y), int(m_x)),
                                       (int(length / 2), stick_width),
                                       int(angle),
                                       0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, OpenPose.colors[i])
            img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)
        return img

    @staticmethod
    def _get_ul_lr(kps):
        not_none_kps = np.array([kp for kp in kps if kp is not None])

        x_max_ind, y_max_ind = np.argmax(not_none_kps, axis=0)
        x_max = not_none_kps[x_max_ind, 0]
        y_max = not_none_kps[y_max_ind, 1]

        x_min_ind, y_min_ind = np.argmin(not_none_kps, axis=0)
        x_min = not_none_kps[x_min_ind, 0]
        y_min = not_none_kps[y_min_ind, 1]

        return x_min, y_min, x_max, y_max

    def _extract_keypoints(self, person_subset, candidate_arr):
        kps = list()
        for i in range(self.n_joints):
            kp_ind = person_subset[i].astype(np.int)
            if kp_ind == -1:
                kps.append(None)
            else:
                kps.append(candidate_arr[kp_ind, 0: 2].astype(np.int))
        return kps

    def draw_pose(self, img):
        org_h, org_w, _ = img.shape
        resized = tf.image.resize_with_pad(img, self.openpose_model.input_h, self.openpose_model.input_w).numpy()
        peaks, subset, candidate = self._inference(resized)

        if not subset.any():
            drawed = None
        else:
            transformed_candidate = self.inverse_transform_kps(org_h, org_w,
                                                               self.openpose_model.input_h,
                                                               self.openpose_model.input_w,
                                                               candidate)
            drawed = self.draw_inverse_transformed_parts(img, peaks, subset, transformed_candidate)
        return drawed

    def _inference(self, img):
        """Img must be of shape (self.box_size // 2, self.box_size // 2), i.e. (184, 184)."""

        paf, masked_heatmap = self.model.predict(np.expand_dims(img, axis=0))
        all_peaks = self._get_peaks(masked_heatmap)
        connection_all, special_k = self._get_connections(paf[0], all_peaks)
        subset, candidate = self._get_subset(all_peaks, special_k, connection_all)
        return all_peaks, subset, candidate

    @staticmethod
    def inverse_transform_kps(org_h, org_w, h, w, candidate):
        kps = candidate[:, 0: 2].astype(np.int)
        scale_factor = np.max([org_h, org_w]) / h
        transformed_candidate = np.zeros((candidate.shape[0], 3))
        if org_h > org_w:
            resized_w = org_w / scale_factor
            border = (w - resized_w) / 2
            for i, kp in enumerate(kps):
                transformed_candidate[i, 0] = scale_factor * (kp[0] - border)
                transformed_candidate[i, 1] = scale_factor * kp[1]
                transformed_candidate[i, 2] = candidate[i, 2]
        else:
            resized_h = org_h / scale_factor
            border = (h - resized_h) / 2
            for i, kp in enumerate(kps):
                transformed_candidate[i, 0] = scale_factor * kp[0]
                transformed_candidate[i, 1] = scale_factor * (kp[1] - border)
                transformed_candidate[i, 2] = candidate[i, 2]
        return transformed_candidate

    @staticmethod
    def draw_inverse_transformed_parts(img, peaks, subset, transformed_candidate):
        valid_indices = subset.flatten().astype(np.int).tolist()[: 18]
        for i in range(18):
            for j in range(len(peaks[i])):
                ind = peaks[i][j][-1]
                if ind in valid_indices:
                    c = transformed_candidate[ind]
                    cv2.circle(img, (int(c[0]), int(c[1])), 4, OpenPose.colors[i], thickness=-1)

        stick_width = 4

        for i in range(17):
            for n in range(len(subset)):
                index = subset[n][np.array(OpenPose.limb_seq[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = img.copy()
                y = transformed_candidate[index.astype(int), 0]
                x = transformed_candidate[index.astype(int), 1]
                m_x = np.mean(x)
                m_y = np.mean(y)
                length = np.sqrt(np.power(x[0] - x[1], 2) + np.power(y[0] - y[1], 2))
                angle = np.degrees(np.arctan2(x[0] - x[1], y[0] - y[1]))
                polygon = cv2.ellipse2Poly((int(m_y), int(m_x)),
                                           (int(length / 2), stick_width),
                                           int(angle),
                                           0,
                                           360,
                                           1)
                cv2.fillConvexPoly(cur_canvas, polygon, OpenPose.colors[i])
                img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)
        return img

    @staticmethod
    def _get_peaks(masked_heatmap):
        ys, xs, channels = np.nonzero(masked_heatmap[0])

        all_peaks = list()
        peak_counter = 0
        for i in range(18):
            indices = np.where(channels == i)[0]
            n_peaks = len(indices)
            peak_inds = range(peak_counter, peak_counter + n_peaks)
            part_peaks = list()
            for j, ind in enumerate(indices):
                x = xs[ind]
                y = ys[ind]
                conf_score = masked_heatmap[0, y, x, i]
                part_peaks.append((x, y, conf_score, peak_inds[j]))
            all_peaks.append(part_peaks)
            peak_counter = peak_counter + n_peaks
        return all_peaks

    def _get_connections(self, paf, all_peaks):
        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(OpenPose.map_idx)):
            score_mid = paf[:, :, [x - 19 for x in OpenPose.map_idx[k]]]
            cand_a = all_peaks[OpenPose.limb_seq[k][0] - 1]
            cand_b = all_peaks[OpenPose.limb_seq[k][1] - 1]
            n_a = len(cand_a)
            n_b = len(cand_b)
            if n_a != 0 and n_b != 0:
                connection_candidate = []
                for i in range(n_a):
                    for j in range(n_b):
                        vec = np.subtract(cand_b[j][:2], cand_a[i][:2])
                        norm = np.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        # failure case when 2 body parts overlaps
                        if norm == 0:
                            continue
                        vec = np.divide(vec, norm)

                        start_end = list(zip(np.linspace(cand_a[i][0], cand_b[j][0], num=mid_num),
                                             np.linspace(cand_a[i][1], cand_b[j][1], num=mid_num)))

                        vec_x = np.array(
                            [score_mid[int(round(start_end[I][1])), int(round(start_end[I][0])), 0]
                             for I in range(len(start_end))])
                        vec_y = np.array(
                            [score_mid[int(round(start_end[I][1])), int(round(start_end[I][0])), 1]
                             for I in range(len(start_end))])

                        score_mid_pts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_mid_pts) / len(score_mid_pts) + min(
                            0.5 * self.openpose_model.input_h / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_mid_pts > self.openpose_model.thre2)[0]) > 0.8 * len(
                            score_mid_pts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior,
                                                         score_with_dist_prior + cand_a[i][2] + cand_b[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [cand_a[i][3], cand_b[j][3], s, i, j]])
                        if len(connection) >= min(n_a, n_b):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])
        return connection_all, special_k

    @staticmethod
    def _get_subset(all_peaks, special_k, connection_all):
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(OpenPose.map_idx)):
            if k not in special_k:
                part_as = connection_all[k][:, 0]
                part_bs = connection_all[k][:, 1]
                index_a, index_b = np.array(OpenPose.limb_seq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][index_a] == part_as[i] or subset[j][index_b] == part_bs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][index_b] != part_bs[i]:
                            subset[j][index_b] = part_bs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][index_b] = part_bs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[index_a] = part_as[i]
                        row[index_b] = part_bs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # delete some rows of subset which has few parts occur
        delete_idx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                delete_idx.append(i)
        subset = np.delete(subset, delete_idx, axis=0)
        return subset, candidate


class FastOpenPoseModel:

    def __init__(self,
                 weights_path,
                 config_path,
                 input_shape,
                 gaussian_filtering):
        self.weights_path = weights_path
        self.config_path = config_path
        self.params, self.model_params = self._read_config()
        self.stride = self.model_params['stride']
        self.pad_value = self.model_params['padValue']
        self.box_size = self.model_params['boxsize']
        self.input_h, self.input_w = input_shape
        self.thre1 = self.params['thre1']
        self.thre2 = self.params['thre2']
        self.model = None
        self.gaussian_filtering = gaussian_filtering

    def load_model(self):
        self.model = self._create_model()
        print('Model loaded successfully')
        return self.model

    def _read_config(self):
        config = ConfigObj(self.config_path)
        param = config['param']
        model_id = param['modelID']
        model = config['models'][model_id]
        model['boxsize'] = int(model['boxsize'])
        model['stride'] = int(model['stride'])
        model['padValue'] = int(model['padValue'])
        param['octave'] = int(param['octave'])
        param['use_gpu'] = int(param['use_gpu'])
        param['starting_range'] = float(param['starting_range'])
        param['ending_range'] = float(param['ending_range'])
        param['scale_search'] = list(map(float, param['scale_search']))
        param['thre1'] = float(param['thre1'])
        param['thre2'] = float(param['thre2'])
        param['thre3'] = float(param['thre3'])
        param['mid_num'] = int(param['mid_num'])
        param['min_num'] = int(param['min_num'])
        param['crop_ratio'] = float(param['crop_ratio'])
        param['bbox_ratio'] = float(param['bbox_ratio'])
        param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])
        return param, model

    def _create_model(self):
        openpose_model = OpenPoseModel()
        openpose_raw = openpose_model.create_model()
        openpose_raw.load_weights(self.weights_path)

        input_tensor = tfkl.Input(shape=(self.input_h, self.input_w, 3))
        x = openpose_raw(input_tensor)
        hm = tf.image.resize(x[1], (self.input_h, self.input_w), 'bicubic')
        paf = tf.image.resize(x[0], (self.input_h, self.input_w), 'bicubic')

        if self.gaussian_filtering:
            gaussian_kernel = self._get_gaussian_kernel()
            depth_wise_gaussian_kernel = tf.expand_dims(
                tf.transpose(tf.keras.backend.repeat(gaussian_kernel, 19), perm=(0, 2, 1)), axis=-1)
            hm = tf.nn.depthwise_conv2d(hm,
                                        depth_wise_gaussian_kernel,
                                        [1, 1, 1, 1],
                                        'SAME')

        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        padded = tf.pad(hm, paddings)

        slice1 = tf.slice(padded, [0, 0, 1, 0], [-1, self.input_h, self.input_w, -1])
        slice2 = tf.slice(padded, [0, 2, 1, 0], [-1, self.input_h, self.input_w, -1])
        slice3 = tf.slice(padded, [0, 1, 0, 0], [-1, self.input_h, self.input_w, -1])
        slice4 = tf.slice(padded, [0, 1, 2, 0], [-1, self.input_h, self.input_w, -1])

        stacked = tf.stack([hm >= slice1, hm >= slice2, hm >= slice3, hm >= slice4, hm >= self.thre1], axis=-1)
        binary_hm = tf.reduce_all(stacked, axis=-1)

        masked_hm = tf.multiply(tf.cast(binary_hm, tf.float32), hm)

        model = tfk.Model(input_tensor, [paf, masked_hm])
        return model

    @staticmethod
    def _get_gaussian_kernel(mean=0, sigma=3):
        size = sigma * 3
        d = tfb.distributions.Normal(mean, sigma)
        vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
        gauss_kernel = tf.einsum('i,j->ij', vals, vals)
        return gauss_kernel


class FeatureExtractor:
    def __init__(self):
        self.points_comb = np.array([[4, 3, 2],
                                     [3, 2, 8],
                                     [8, 2, 5],
                                     [2, 5, 11],
                                     [7, 6, 5],
                                     [6, 5, 11],
                                     [2, 8, 11],
                                     [5, 11, 8],
                                     [8, 9, 10],
                                     [11, 12, 13],
                                     [9, 8, 11],
                                     [12, 11, 8],
                                     [2, 1, 5],
                                     [16, 2, 5],
                                     [17, 5, 2],
                                     [2, 16, 1],
                                     [1, 17, 5]])

    def generate_features(self, keypoints):
        features = list()
        for comb in self.points_comb:
            feature = None

            a = keypoints[comb[0]]
            b = keypoints[comb[1]]
            c = keypoints[comb[2]]

            if (a is not None) and (b is not None) and (c is not None):
                feature = self._compute_angle(np.array(a),
                                              np.array(b),
                                              np.array(c))
            features.append(feature)
        return np.array(features)

    @staticmethod
    def _compute_angle(a, b, c):
        """Computes angle on point 'b'."""
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / ((np.linalg.norm(ba) * np.linalg.norm(bc)) + 0.0001)
        angle = np.arccos(cosine_angle)
        angle = np.degrees(angle)
        return angle


def timing(func):
    def inner(*args, **kwargs):
        t = time()
        ret = func(*args, **kwargs)
        print('{} execution time: {} s'.format(func.__name__, time() - t))
        return ret

    return inner
