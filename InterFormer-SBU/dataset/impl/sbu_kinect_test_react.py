import os
from pathlib import Path
import numpy as np


from dataset.dataset import Dataset, HyperParameterSet
from dataset.augmentation import AugRandomScale, AugRandomTranslation
from dataset.impl.lowlevel import Sample, LowLevelDataset
from utils.logger import log


# ----------------------------------------------------------------------------------------------------------------------
class DatasetSBUKinect(Dataset):
    def __init__(self, root="data/sbu/SBU-Kinect-Interaction", num_synth=0):
        super(DatasetSBUKinect, self).__init__("SBU-Kinect", root, num_synth)

    def _load_underlying_dataset(self):
        self.underlying_dataset = self._load_sbu_kinect_interaction()
        self.num_features = 45  # 3D world coordinates joints of two people (15 joints x 3 dimensions).
                                # Each row is one person. Every two rows make up one frame.
        self.num_folds = 5      # This dataset has 5 folds

    def get_hyperparameter_set(self):
        return HyperParameterSet(learning_rate=0.001,
                                 batch_size=64,
                                 weight_decay=0,
                                 num_epochs=10)#200 by default

    def _get_augmenters(self, random_seed):
        return [
            AugRandomScale(3, self.num_synth, random_seed, 0.7, 1.3),
            AugRandomTranslation(3, self.num_synth, random_seed, -1, 1),
        ]

    def _load_sbu_kinect_interaction(self, unnormalize=True, verbose=False):
        """
        Loads the SBU Kinect Interactions dataset. We unnormalize the raw data using the equations
        that are provided in the dataset's documentation.
        """


        # Each action's label
        LABELS = {"01": "Approaching",
                  "03": "Kicking",
                  "04": "Pushing",
                  "05": "ShakingHands",
                  "07": "Exchanging",
                  "08": "Punching"}

        # Pre-set 5-fold cross validations from dataset's README
        # FOLD[i] means train on every other fold, test on fold i
        FOLDS = [["s00s00"],
            ["s01s02", "s03s04", "s05s02", "s06s04"],
            ["s02s03", "s02s07", "s03s05", "s05s03"],
            ["s01s03", "s01s07", "s07s01", "s07s03"],
            ["s02s01", "s03s02", "s06s03","s02s06"],
            ["s04s02", "s04s03", "s06s02","s03s06","s04s06"]
        ]

        # Number of folds
        FOLD_CNT = len(FOLDS)

        # Number of joints
        JOINT_CNT = 15

        # Using 5-fold cross validation as the predefined test
        # (e.g. train[0] test[0] mean test on FOLD[0], train on everything else)
        train_indices = [[] for i in range(FOLD_CNT)]
        test_indices = [[] for i in range(FOLD_CNT)]
        samples = []

        for fname in Path(self.root).glob('**/*.txt'):
            fname = str(fname)

            # Determine sample properties
            subject, label, example = fname.replace(self.root, '').split('\\')[-4:-1]

            if verbose:
                log("load: {}, label {}, subject {}, example {}".format(fname,
                                                                        label,
                                                                        subject,
                                                                        example))

            # Now read the actual file
            with open(fname) as f:
                lines = f.read().splitlines()

            pts = []
            body_pts = {0: []}  # body index -> list of points in the entire sequence
            framecount = len(lines)

            for idx, line in enumerate(lines):
                line = line.split(',')[1:]  # Skip the frame number

                for body in range(1):
                    pt = np.zeros(3 * JOINT_CNT, dtype=np.float32)

                    # Read the (x, y, z) position of the joint of each person (2 people in each frame)
                    for i in range(JOINT_CNT):
                        pt[3 * i + 0] = float(line[(3 * body * JOINT_CNT) + (3 * i + 0)])
                        pt[3 * i + 1] = float(line[(3 * body * JOINT_CNT) + (3 * i + 1)])
                        pt[3 * i + 2] = float(line[(3 * body * JOINT_CNT) + (3 * i + 2)])

                        # Unnormalize the joint positions if requested (formula's from dataset's README file)
                        if unnormalize:
                            pt[3 * i + 0] = (1280 - (pt[3 * i + 0] * 2560)) / 1000
                            pt[3 * i + 1] = (960 - (pt[3 * i + 1] * 1920)) / 1000
                            pt[3 * i + 2] = (pt[3 * i + 2] * 10000 / 7.8125) / 1000

                    body_pts[body] += [pt]

            # Sanity check
            for b_idx in range(1):
                assert len(body_pts[b_idx]) == framecount

            # Sort bodies based on activity (high-activity first)
            bodies_by_activity = sorted(body_pts.items(),
                                        key=lambda item: DatasetSBUKinect._calculate_motion(np.asarray(item[1]),
                                                                                            JOINT_CNT), reverse=True)

            for f in range(framecount):
                for b_idx, bodys_frames in bodies_by_activity:
                    pt = bodys_frames[f]
                    pts += [pt]

            # Make a sample
            samples += [Sample(pts, LABELS[label], subject, fname)]

            # Add the index to train/test indices for each fold
            s_idx = len(samples) - 1

            for fold_idx in range(FOLD_CNT):
                fold = FOLDS[fold_idx]

                if subject in fold:
                    # Add the instance as a TESTING instance to this fold
                    test_indices[fold_idx] += [s_idx]

                    # For all other folds, this guy would be a TRAINING instance
                    for other_idx in range(FOLD_CNT):
                        if fold_idx == other_idx:
                            continue

                        train_indices[other_idx] += [s_idx]

        # k-fold sanity check
        for fold_idx in range(FOLD_CNT):
            assert len(train_indices[fold_idx]) + len(test_indices[fold_idx]) == len(samples)
            # Ensure there is no intersection between training/test indices
            assert len(set(train_indices[fold_idx]).intersection(test_indices[fold_idx])) == 0

        return LowLevelDataset(samples, train_indices, test_indices)


    @staticmethod
    def _calculate_motion(np_pts, num_joints):
        """
        Measures the motion of a sample
        """
        total_motion = 0
        motion_per_joint = []

        # Slice along all frames for each joint, then calculate the Euclidean length of that sequence
        for j in range(num_joints):
            joint_col_start = 3 * j
            joint_col_end = 3 * (j + 1)

            pts = np_pts[:, joint_col_start:joint_col_end]

            displacement = DatasetSBUKinect._series_len(pts)
            total_motion += displacement
            motion_per_joint += [displacement]

        assert len(motion_per_joint) == num_joints

        return total_motion, motion_per_joint

    @staticmethod
    def _series_len(pts):
        """
        Computes the path length of a sample
        """
        ret = 0.0

        for idx in range(1, len(pts)):
            ret += np.linalg.norm(pts[idx] - pts[idx - 1])

        return ret
