import copy
import numpy as np
import random


# ----------------------------------------------------------------------------------------------------------------------
class LowLevelDataset(object):
    """
    Represents a "low level" dataset of samples. This class deals with the nitty-gritty
    details of loading the data, how the train/tests are defined and all other details of the
    dataset that may not be necessarily relevant to PyTorch.
    This is meant to be a proxy between Python/Numpy types and PyTorch.
    """
    def __init__(self, samples, train_indices=None, test_indices=None):
        """ """
        label_to_samples = {}
        for sample in samples:
            label = sample.label
            if not label in label_to_samples:
                label_to_samples[label] = []
            label_to_samples[label] += [sample]
        self.samples = samples
        self.class_labels = sorted(label_to_samples.keys())
        # Custom training/testing indices
        self.train_indices = train_indices
        self.test_indices = test_indices

    def __len__(self):
        return len(self.samples)

    def get_indices_for_fold(self, fold_idx, shuffle, random_seed):
        """
        Gets the training/testing indices of the data instances associated with the specified fold.
        """

        train_indices = copy.deepcopy(self.train_indices[fold_idx])

        if shuffle:
            random.Random(random_seed).shuffle(train_indices)

        result_train = [self.samples[i] for i in train_indices]
        result_test = [self.samples[i] for i in self.test_indices[fold_idx]]

        return result_train, result_test


# ----------------------------------------------------------------------------------------------------------------------
class Sample(object):
    """
    Represents one sample and its label in a low-level dataset.
    """

    def __init__(self, pts, label, subject, path):
        """ """
        self.pts = pts
        self.label = label
        self.subject = subject
        self.path = path

    def __eq__(self, other):
        """
        Equality check of two low-level samples that is the foundation of all our sanity checks.
        """
        return self.path == other.path

    def to_numpy(self):
        return np.array(self.pts, dtype=np.float32), self.label
