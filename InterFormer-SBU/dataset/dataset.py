import sys
import random
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import Sequence
from torch.utils.data.sampler import Iterator

from utils.logger import log


# ----------------------------------------------------------------------------------------------------------------------
class HyperParameterSet:
    """
    Defines the hyper parameters that need to be used with a particular dataset
    """
    def __init__(self,
                 learning_rate,
                 batch_size,
                 weight_decay,
                 num_epochs):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

    def __str__(self):
        return "\n".join([
            "Learning rate: " + str(self.learning_rate),
            "Batch size: " + str(self.batch_size),
            "Weight decay: " + str(self.weight_decay),
            "Num epochs: " + str(self.num_epochs),
            ])


# ----------------------------------------------------------------------------------------------------------------------
class PadCollate:
    """
    A DataLoader collation function that zero-pads to the length of the longest sequence in a batch
    """

    def __call__(self, batch):
        # Sort based on the length of each sequence
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        xs = [x[0] for x in sorted_batch]
        ys = [x[1] for x in sorted_batch]
        xs_padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys)
        lengths = torch.LongTensor([len(x) for x in xs])
        return xs_padded, lengths, ys


# ----------------------------------------------------------------------------------------------------------------------

class SubsetRandomSampler_v2(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator
        self.idx_sample = indices

    def __iter__(self) -> Iterator[int]:
        k=0
        for i in torch.randperm(len(self.indices), generator=self.generator):
            self.idx_sample[k]=i
            k+=1
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)


class Dataset(data.Dataset):
    """
    An abstract class representing a general dataset that has training/testing splits.
    Actual data loading mechanisms are expected to subclass and implement this.
    Objects of this type load training/testing samples into a "cache" of PyTorch tensors.
    This is meant to represent types that are closer to PyTorch.
    """

    def __init__(self, name, root, num_synth):
        self.name = name
        self.root = root
        self.num_synth = num_synth

        # The list that stores all the examples. For training/testing with PyTorch's DataLoader we
        # index into this cache, which contains post-processed (i.e. normalized, augmented, etc) cache
        # of data
        self._cache = None

        self.generator = None  # Used for training cases where a generator is needed

        # The actual dataset that is loaded from the files
        self.underlying_dataset = None

        # Dataset specific things. These are populated once the dataset is loaded
        self.num_classes = None
        self.num_features = None
        self.num_samples = None
        self.num_folds = None

        # Mapping dictionaries
        self.class_to_idx = {}
        self.idx_to_class = {}

        log("Reading the '{}' dataset...".format(self.name))
        self._load_dataset()

    def get_hyperparameter_set(self):
        """
        Returns a set of hyperparamters that work well for this dataset
        """

        raise NotImplementedError

    def __getitem__(self, index):
        return self._cache[index]

    def __len__(self):
        return len(self._cache)

    def __str__(self):
        return "\n".join([
            "Dataset: {}".format(self.name),
            "Classes: {}".format(self.idx_to_class),
            "Num samples: {}".format(self.num_samples),
            "Num synth: {}".format(self.num_synth),
            str(self.get_hyperparameter_set()),
            "\n"
        ])

    def _load_dataset(self):
        self._load_underlying_dataset()  # Loads the specific underlying dataset

        self.num_classes = len(self.underlying_dataset.class_labels)
        self.num_samples = len(self.underlying_dataset)

        self.class_to_idx = {ch: i for i, ch in enumerate(self.underlying_dataset.class_labels)}
        self.idx_to_class = {i: ch for i, ch in enumerate(self.underlying_dataset.class_labels)}

    def _load_underlying_dataset(self):
        """
        Loads the underlying dataset that this class would interface with
        """
        raise NotImplementedError

    def _get_augmenters(self, random_seed):
        """
        Returns the set of data augmentors that work well for this dataset
        """
        raise NotImplementedError

    def get_data_loaders(self, fold_idx, shuffle=True, random_seed=1223, normalize=True,num_worker=4):
        """
        Returns the torch.nn.data.DataLoader instances for training/testing on this dataset
        """
        batch_size = self.get_hyperparameter_set().batch_size

        train_sampler, test_sampler = self._get_train_test_sampler(fold_idx, shuffle, random_seed, normalize)
        train_loader = DataLoader(dataset=self,
                                  sampler=train_sampler,
                                  collate_fn=PadCollate(),
                                  batch_size=batch_size,
                                  pin_memory=True,
                                  num_workers=num_worker)
        test_loader = DataLoader(dataset=self,
                                 sampler=test_sampler,
                                 collate_fn=PadCollate(),
                                 batch_size=batch_size,
                                 pin_memory=True,
                                 num_workers=num_worker)

        return train_loader, test_loader

    def _get_train_test_sampler(self, fold_idx, shuffle, random_seed=None, normalize=False):
        """
        Creates the training/testing samplers for this dataset.
        This function populates self._cache and computes z-score normalization factors (if requested)
        """

        # Get the low level dataset's indices
        train, test = self.underlying_dataset.get_indices_for_fold(fold_idx, shuffle, random_seed)

        # Fill the cache and data indices, possibly augment the data and z-score normalize everything
        train_indices = []
        test_indices = []
        self._cache = []

        if self.num_synth == 0:
            generate_synth = False
            #log("Filling cache...")
        else:
            generate_synth = True
            #log("Filling cache and generating {} synthetic samples...".format(self.num_synth))

        aggregate_data = []

        if generate_synth:
            augs = self._get_augmenters(random_seed)

            if self.num_synth > 0 and augs is None or len(augs) == 0:
                raise Exception('Augmentation requested but this dataset is not producing valid data augmentors!')

        # Training
        for idx, sample in enumerate(train):
            pts, class_name = sample.to_numpy()

            # Convert to PyTorch tensor
            result = torch.from_numpy(pts)
            label = torch.from_numpy(np.asarray(self.class_to_idx[class_name]))

            self._cache += [(result, label, False, sample.path)]
            train_indices += [len(self) - 1]

            if normalize:
                aggregate_data.append(pts)

            # Generate synthetic samples if needed

            if generate_synth:
                for a_idx, aug_a in enumerate(augs):
                    synth_samples_a = aug_a.generate_samples(pts)

                    if synth_samples_a is not None:
                        # Add each sample to the training set
                        for synth_pts_a in synth_samples_a:

                            # Convert to PyTorch tensor
                            result = torch.from_numpy(synth_pts_a)
                            label = label.clone()

                            self._cache += [(result, label, True, sample.path)]

                            if normalize:
                                aggregate_data.append(synth_pts_a)

                            # Add the synthetic sample to the training set as well
                            train_indices += [len(self) - 1]

        # Testing
        for sample in test:
            pts, class_name = sample.to_numpy()

            # Convert to PyTorch tensor
            result = torch.from_numpy(pts)
            label = torch.from_numpy(np.asarray(self.class_to_idx[class_name]))

            self._cache += [(result, label, False, sample.path)]
            test_indices += [len(self) - 1]

        # Shuffle the training indices
        if shuffle:
            random.Random(random_seed).shuffle(train_indices)

        # Do appropriate z-score normalization if requested
        if normalize:
            # Calculate the mean/stdev
            aggregate_data = np.concatenate(aggregate_data)
            avg = np.average(aggregate_data, axis=0)
            std = np.std(aggregate_data, axis=0, dtype=np.float32)
            std[std == 0] = 1

            indices = list(range(len(self)))  # To account for synthetic samples that were added too

            for i in indices:
                sample, label, is_synth, sample_path = self[i]
                pt = sample.numpy()
                new_pt = (pt - avg) / std
                self._cache[i] = (torch.from_numpy(new_pt), label, is_synth, sample_path)

        # Do some extremely important sanity checks:
        # 1) Ensure no overlap between train/test set after everything is done:
        if (len(set([self._cache[idx][3] for idx in train_indices]).intersection(
                [self._cache[idx][3] for idx in test_indices])) != 0):
            raise Exception("The intersection of training and testing set is not the empty set!!!")

        # 2) Ensure that there are no synthetic samples in the testing set
        if any([self._cache[idx][2] for idx in test_indices]):
            raise Exception("At least one of the data in the test set is synthetic!!!")

        # Return the samplers as well as the normalizing factors
        return SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices)
