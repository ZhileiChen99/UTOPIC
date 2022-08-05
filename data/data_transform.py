import math
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
import torch
import torch.utils.data

from utils.random import uniform_2_sphere
import utils.se3 as se3
import utils.so3 as so3

from sklearn.neighbors import NearestNeighbors


class SplitSourceRef:
    """Clones the point cloud into separate source and reference point clouds"""

    def __call__(self, sample: Dict):
        sample['points_raw'] = sample.pop('points')
        if isinstance(sample['points_raw'], torch.Tensor):
            sample['points_src'] = sample['points_raw'].detach()
            sample['points_ref'] = sample['points_raw'].detach()
        else:  # is numpy
            sample['points_src'] = sample['points_raw'].copy()
            sample['points_ref'] = sample['points_raw'].copy()

        return sample


class Resampler:
    def __init__(self, num: int):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'] = self._resample(sample['points'], self.num)
        else:
            if 'crop_proportion' not in sample:
                src_size, ref_size = self.num, self.num
            elif len(sample['crop_proportion']) == 1:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = self.num
            elif len(sample['crop_proportion']) == 2:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = math.ceil(sample['crop_proportion'][1] * self.num)
            else:
                raise ValueError('Crop proportion must have 1 or 2 elements')

            if 'tranflag' in sample:
                sample['points_src'] = self._resample(sample['points_src'], src_size)
                sample['points_ref'] = self._resample(sample['points_ref'], ref_size)
            else:
                sample['points_src'] = self._resample(sample['points_src'], src_size)
                sample['points_ref'] = sample['points_src']

        return sample

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]


class FixedResampler(Resampler):
    """Fixed resampling to always choose the first N points.
    Always deterministic regardless of whether the deterministic flag has been set
    """

    @staticmethod
    def _resample(points, k):
        multiple = k // points.shape[0]
        remainder = k % points.shape[0]

        resampled = np.concatenate((np.tile(points, (multiple, 1)), points[:remainder, :]), axis=0)
        return resampled


class RandomJitter:
    """ generate perturbations """

    def __init__(self, scale=0.01, clip=0.05):
        self.scale = scale
        self.clip = clip

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0, scale=self.scale, size=(pts.shape[0], 3)),
                        a_min=-self.clip, a_max=self.clip)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def __call__(self, sample):

        if 'points' in sample:
            sample['points'] = self.jitter(sample['points'])
        else:
            sample['points_src'] = self.jitter(sample['points_src'])
            sample['points_ref'] = self.jitter(sample['points_ref'])

        return sample


class RandomCrop:
    """Randomly crops the *source* point cloud, approximately retaining half the points

    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """

    def __init__(self, p_keep: List = None):
        if p_keep is None:
            p_keep = [0.7, 0.7]  # Crop both clouds to 70%
        self.p_keep = np.array(p_keep, dtype=np.float32)

    @staticmethod
    def crop(points, p_keep):
        rand_xyz = uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :]

    def __call__(self, sample):

        sample['crop_proportion'] = self.p_keep
        if np.all(self.p_keep == 1.0):
            return sample  # No need crop

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if len(self.p_keep) == 1:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0])
        else:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0])
            sample['points_ref'] = self.crop(sample['points_ref'], self.p_keep[1])
        return sample


class RandomCropinv:
    """Randomly crops the *source* point cloud, approximately retaining half the points

    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """

    def __init__(self, p_keep: List = None):
        if p_keep is None:
            p_keep = [0.7, 0.7]  # Crop both clouds to 70%
        self.p_keep = np.array(p_keep, dtype=np.float32)

    @staticmethod
    def crop(points, p_keep):
        rand_xyz = uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane < 0
        else:
            mask = dist_from_plane < np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :]

    def __call__(self, sample):

        sample['crop_proportion'] = self.p_keep
        if np.all(self.p_keep == 1.0):
            return sample  # No need crop

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if len(self.p_keep) == 1:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0])
        else:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0])
            sample['points_ref'] = self.crop(sample['points_ref'], self.p_keep[1])
        return sample


class RandomTransformSE3:
    def __init__(self, rot_mag: float = 180.0, trans_mag: float = 1.0, random_mag: bool = False):
        """Applies a random rigid transformation to the source point cloud

        Args:
            rot_mag (float): Maximum rotation in degrees
            trans_mag (float): Maximum translation T. Random translation will
              be in the range [-X,X] in each axis
            random_mag (bool): If true, will randomize the maximum rotation, i.e. will bias towards small
                               perturbations
        """
        self._rot_mag = rot_mag
        self._trans_mag = trans_mag
        self._random_mag = random_mag

    def generate_transform(self):
        """Generate a random SE3 transformation (3, 4) """

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        rand_rot = special_ortho_group.rvs(3)
        axis_angle = Rotation.as_rotvec(Rotation.from_dcm(rand_rot))
        axis_angle *= rot_mag / 180.0
        rand_rot = Rotation.from_rotvec(axis_angle).as_dcm()

        # Generate translation
        rand_trans = np.random.uniform(-trans_mag, trans_mag, 3)
        rand_SE3 = np.concatenate((rand_rot, rand_trans[:, None]), axis=1).astype(np.float32)

        return rand_SE3

    def apply_transform(self, p0, transform_mat):
        p1 = se3.transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        igt = transform_mat
        gt = se3.inverse(igt)

        return p1, gt, igt

    def transform(self, tensor):
        transform_mat = self.generate_transform()
        return self.apply_transform(tensor, transform_mat)

    def __call__(self, sample):
        sample['tranflag'] = True

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'], _, _ = self.transform(sample['points'])
        else:
            src_transformed, transform_r_s, transform_s_r = self.transform(sample['points_src'])
            sample['transform_gt'] = transform_r_s  # Apply to source to get reference
            sample['points_src'] = src_transformed

            sample['points_ref_raw'] = sample['points_raw']
            sample['points_src_raw'], _, _ = self.apply_transform(sample['points_raw'], transform_s_r)

        return sample


# noinspection PyPep8Naming
class RandomTransformSE3_euler(RandomTransformSE3):
    """Same as RandomTransformSE3, but rotates using euler angle rotations

    This transformation is consistent to Deep Closest Point but does not
    generate uniform rotations

    """

    def generate_transform(self):

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        anglex = np.random.uniform() * np.pi * rot_mag / 180.0
        angley = np.random.uniform() * np.pi * rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-trans_mag, trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        return rand_SE3


class RandomRotatorZ(RandomTransformSE3):
    """Applies a random z-rotation to the source point cloud"""

    def __init__(self):
        super().__init__(rot_mag=360)

    def generate_transform(self):
        """Generate a random SE3 transformation (3, 4) """

        rand_rot_deg = np.random.random() * self._rot_mag
        rand_rot = Rotation.from_euler('z', rand_rot_deg, degrees=True).as_dcm()
        rand_SE3 = np.pad(rand_rot, ((0, 0), (0, 1)), mode='constant').astype(np.float32)

        return rand_SE3


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


class ShufflePoints:
    """Shuffles the order of the points"""

    def __call__(self, sample):
        if 'points' in sample:
            sample['points'] = np.random.permutation(sample['points'])
        else:
            refperm = np.random.permutation(sample['points_ref'].shape[0])
            srcperm = np.random.permutation(sample['points_src'].shape[0])
            sample['points_ref'] = sample['points_ref'][refperm, :]
            sample['points_src'] = sample['points_src'][srcperm, :]
            if 'jitterflag' in sample or 'cropflag' in sample:
                perm_mat = np.zeros((sample['points_src'].shape[0], sample['points_ref'].shape[0]))
                inlier_src = np.zeros((sample['points_src'].shape[0], 1))
                inlier_ref = np.zeros((sample['points_ref'].shape[0], 1))
                points_src_transform = se3.transform(sample['transform_gt'], sample['points_src'][:, :3])
                points_ref = sample['points_ref'][:, :3]
                dist_s2et, indx_s2et = nearest_neighbor(points_src_transform, points_ref)
                dist_t2es, indx_t2es = nearest_neighbor(points_ref, points_src_transform)
                padtype = 3
                padth = 0.05
                if padtype == 1:
                    for row_i in range(sample['points_src'].shape[0]):
                        if indx_t2es[indx_s2et[row_i]] == row_i and dist_s2et[row_i] < padth:
                            perm_mat[row_i, indx_s2et[row_i]] = 1
                elif padtype == 2:
                    for row_i in range(sample['points_src'].shape[0]):
                        if dist_s2et[row_i] < padth:
                            perm_mat[row_i, indx_s2et[row_i]] = 1
                    for col_i in range(sample['points_ref'].shape[0]):
                        if dist_t2es[col_i] < padth:
                            perm_mat[indx_t2es[col_i], col_i] = 1
                elif padtype == 3:
                    for row_i in range(sample['points_src'].shape[0]):
                        if indx_t2es[indx_s2et[row_i]] == row_i and dist_s2et[row_i] < padth:
                            perm_mat[row_i, indx_s2et[row_i]] = 1
                    for row_i in range(sample['points_src'].shape[0]):
                        if np.sum(perm_mat[row_i, :]) == 0 \
                                and np.sum(perm_mat[:, indx_s2et[row_i]]) == 0 \
                                and dist_s2et[row_i] < padth:
                            perm_mat[row_i, indx_s2et[row_i]] = 1
                    for col_i in range(sample['points_ref'].shape[0]):
                        if np.sum(perm_mat[:, col_i]) == 0 \
                                and np.sum(perm_mat[indx_t2es[col_i], :]) == 0 \
                                and dist_t2es[col_i] < padth:
                            perm_mat[indx_t2es[col_i], col_i] = 1
                    outlier_src_ind = np.where(np.sum(perm_mat, axis=1) == 0)[0]
                    outlier_ref_ind = np.where(np.sum(perm_mat, axis=0) == 0)[0]
                    points_src_transform_rest = points_src_transform[outlier_src_ind]
                    points_ref_rest = points_ref[outlier_ref_ind]
                    if points_src_transform_rest.shape[0] > 0 and points_ref_rest.shape[0] > 0:
                        dist_s2et, indx_s2et = nearest_neighbor(points_src_transform_rest, points_ref_rest)
                        dist_t2es, indx_t2es = nearest_neighbor(points_ref_rest, points_src_transform_rest)
                        for row_i in range(points_src_transform_rest.shape[0]):
                            if indx_t2es[indx_s2et[row_i]] == row_i and dist_s2et[row_i] < padth * 2:
                                perm_mat[outlier_src_ind[row_i], outlier_ref_ind[indx_s2et[row_i]]] = 1
                inlier_src_ind = np.where(np.sum(perm_mat, axis=1))[0]
                inlier_ref_ind = np.where(np.sum(perm_mat, axis=0))[0]
                inlier_src[inlier_src_ind] = 1
                inlier_ref[inlier_ref_ind] = 1
                sample['perm_mat'] = perm_mat
                sample['src_overlap_gt'] = inlier_src
                sample['ref_overlap_gt'] = inlier_ref
            else:
                perm_mat = np.zeros((sample['points_src'].shape[0], sample['points_ref'].shape[0]))
                srcpermsort = np.argsort(srcperm)
                refpermsort = np.argsort(refperm)
                for i, j in zip(srcpermsort, refpermsort):
                    perm_mat[i, j] = 1
                sample['perm_mat'] = perm_mat
                sample['src_overlap_gt'] = np.ones((sample['points_src'].shape[0], 1))
                sample['ref_overlap_gt'] = np.ones((sample['points_ref'].shape[0], 1))

        return sample


class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""

    def __call__(self, sample):
        sample['deterministic'] = True
        return sample


class SetJitterFlag:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""

    def __call__(self, sample):
        sample['jitterflag'] = True
        return sample


class SetCropFlag:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""

    def __call__(self, sample):
        sample['cropflag'] = True
        return sample


class Dict2DcpList:
    """Converts dictionary of tensors into a list of tensors compatible with Deep Closest Point"""

    def __call__(self, sample):
        target = sample['points_src'][:, :3].transpose().copy()
        src = sample['points_ref'][:, :3].transpose().copy()

        rotation_ab = sample['transform_gt'][:3, :3].transpose().copy()
        translation_ab = -rotation_ab @ sample['transform_gt'][:3, 3].copy()

        rotation_ba = sample['transform_gt'][:3, :3].copy()
        translation_ba = sample['transform_gt'][:3, 3].copy()

        euler_ab = Rotation.from_dcm(rotation_ab).as_euler('zyx').copy()
        euler_ba = Rotation.from_dcm(rotation_ba).as_euler('xyz').copy()

        return src, target, \
               rotation_ab, translation_ab, rotation_ba, translation_ba, \
               euler_ab, euler_ba


class Dict2PointnetLKList:
    """Converts dictionary of tensors into a list of tensors compatible with PointNet LK"""

    def __call__(self, sample):

        if 'points' in sample:
            # Train Classifier (pretraining)
            return sample['points'][:, :3], sample['label']
        else:
            # Train PointNetLK
            transform_gt_4x4 = np.concatenate([sample['transform_gt'],
                                               np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)], axis=0)
            return sample['points_src'][:, :3], sample['points_ref'][:, :3], transform_gt_4x4


class PRNetTorchOverlapRatio:
    def __init__(self, crop_num_points, rot_mag, trans_mag,
                 noise_std=0.01, clip=0.05, add_noise=True, overlap_ratio=0.69):
        self.crop_num_points = crop_num_points
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.noise_std = noise_std
        self.clip = clip
        self.add_noise = add_noise
        self.overlap_ratio = overlap_ratio

    def apply_transform(self, p0, transform_mat):
        p1 = se3.transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        gt = transform_mat

        return p1, gt

    def jitter(self, pts):
        noise = np.clip(np.random.normal(0.0, scale=self.noise_std, size=(pts.shape[0], 3)), a_min=-self.clip,
                        a_max=self.clip)
        noise = torch.from_numpy(noise).to(pts.device)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def knn(self, pts, random_pt, k1, k2):
        random_pt = torch.from_numpy(random_pt).to(pts.device)
        distance = torch.sum((pts - random_pt) ** 2, dim=1)
        idx1 = distance.topk(k=k1, dim=0, largest=False)[1]  # (batch_size, num_points, k)
        idx2 = distance.topk(k=k2, dim=0, largest=True)[1]
        return idx1, idx2

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        src = sample["points_src"]
        ref = sample["points_ref"]

        # Crop and sample
        src = torch.from_numpy(src)
        ref = torch.from_numpy(ref)
        random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
        # pdb.set_trace()
        src_idx1, src_idx2 = self.knn(src, random_p1, k1=self.crop_num_points, k2=2048 - self.crop_num_points)
        ref_idx1, ref_idx2 = self.knn(ref, random_p1, k1=self.crop_num_points, k2=2048 - self.crop_num_points)

        k1_idx = ref_idx1[np.random.randint(ref_idx1.size()[0])]
        k1 = ref[k1_idx, :]
        k2_idx = ref_idx1[0]
        k2 = ref[k2_idx, :]

        distance = torch.sum((ref[ref_idx1, :] - k1) ** 2, dim=1)
        overlap_idx = distance.topk(k=int(self.crop_num_points * self.overlap_ratio), dim=0, largest=False)[1]
        k1_points = ref[ref_idx1, :][overlap_idx, :]
        distance = torch.sum((ref[ref_idx2, :] - k2) ** 2, dim=1)
        nonoverlap_idx = distance.topk(k=self.crop_num_points - int(self.crop_num_points * self.overlap_ratio), dim=0, largest=False)[1]
        k2_points = ref[ref_idx2, :][nonoverlap_idx, :]
        ref = torch.cat((k1_points, k2_points), dim=0)
        src = src[src_idx1, :]

        # Generate rigid transform
        anglex = np.random.uniform() * np.pi * self.rot_mag / 180.0
        angley = np.random.uniform() * np.pi * self.rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * self.rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-self.trans_mag, self.trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        ref, transform_s_r = self.apply_transform(ref, rand_SE3)
        # Apply to source to get reference
        sample["transform_gt"] = transform_s_r

        sample['points_src_raw'] = sample['points_raw']
        sample['points_ref_raw'], _ = self.apply_transform(sample['points_raw'], transform_s_r)

        # add noise
        if self.add_noise:
            sample["points_src"] = self.jitter(src)
            sample["points_ref"] = self.jitter(ref)
        else:
            sample["points_src"] = src
            sample["points_ref"] = ref

        if sample["points_src"].size()[0] == 1:
            sample["points_src"] = sample["points_src"].squeeze(0)
            sample["points_ref"] = sample["points_ref"].squeeze(0)

        return sample
