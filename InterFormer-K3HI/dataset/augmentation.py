import random
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
class DataAugmenter(object):
    """
    Defines the interface for data augmentation
    """
    def __init__(self, dimensionality, num_synth, random_seed):
        self.dimensionality = dimensionality
        self.num_synth = num_synth
        self.random_seed = random_seed
        self.random = random.Random(random_seed)

    def generate_samples(self, pts):
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------------------------------
class AugRandomScale(DataAugmenter):
    """
    Performs random scaling on a sample with the specified factors
    """
    def __init__(self, dimensionality, num_synth, random_seed, factor_start, factor_end):
        super(AugRandomScale, self).__init__(dimensionality, num_synth, random_seed)
        self.factor_start = factor_start
        self.factor_end = factor_end

    def generate_samples(self, pts):
        ret = []
        orig_cols = pts.shape[1]
        reshaped = pts.reshape(-1, self.dimensionality)

        for i in range(self.num_synth):
            while True:
                rnd = [self.random.uniform(self.factor_start, self.factor_end) for i in range(self.dimensionality)]
                is_good = False

                for d in range(self.dimensionality):
                    is_good = is_good or rnd[d] != 1

                    if is_good:
                        break
                if is_good:
                    rnd = np.asarray(rnd, dtype=np.float32)
                    break

            synth_pts = rnd * reshaped
            synth_pts = synth_pts.reshape(-1, orig_cols)
            ret += [synth_pts]

        return ret


# ----------------------------------------------------------------------------------------------------------------------
class AugRandomTranslation(DataAugmenter):
    """
        Performs random translation on a sample with the specified factors
    """
    def __init__(self, dimensionality, num_synth, random_seed, factor_start, factor_end):
        super(AugRandomTranslation, self).__init__(dimensionality, num_synth, random_seed)
        self.factor_start = factor_start
        self.factor_end = factor_end

    def generate_samples(self, pts):
        ret = []
        orig_cols = pts.shape[1]
        reshaped = pts.reshape(-1, self.dimensionality)

        for i in range(self.num_synth):
            while True:
                rnd = [self.random.uniform(self.factor_start, self.factor_end) for i in range(self.dimensionality)]
                is_good = False

                for d in range(self.dimensionality):
                    is_good = is_good or rnd[d] != 0

                    if is_good:
                        break

                if is_good:
                    rnd = np.asarray(rnd, dtype=np.float32)
                    break

            # Tile to the same dim as the input
            t = np.tile(rnd, [reshaped.shape[0], 1])
            synth_pts = reshaped + t
            synth_pts = synth_pts.reshape(-1, orig_cols)
            ret += [synth_pts]

        return ret
