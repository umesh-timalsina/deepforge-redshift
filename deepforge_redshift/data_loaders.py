import numpy as np
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.utils import Sequence

from deepforge_redshift.utils import get_logger

MAX_REDSHIFT_VALUES = 0.4
MAX_DERED_PETRO_MAG = 17.8
REDSHIFT_KEY = "z"
DEREDENED_PETRO_KEY = "dered_petro_r"
EBV_KEY = "EBV"


class DataSetSampler:
    """Sampler for the dataset
    This class is used to load the Dataset(~53GB) or a mmap,
    sample it and return the indexes according to the values.

    Parameters
    ----------
    seed : int, default = None
        random seed for numpy
    cube : np.ndarray, default=None
        the datacube
    labels : np.ndarray, default=None
        the labels array
    test_size: float, default=0.2
        The test size of the dataset

    Attributes
    ----------
    cube : np.ndarray
        The numpy array of the base dataset
    labels : np.ndarray
        The base directory for the dataset
    train_indices : np.ndarray
        The intersection of the indexes in the dataset with
        redshifts in the desired deredened petro mags
    """

    def __init__(self, cube, labels, seed=42, test_size=0.2):
        self.logger = get_logger(self.__class__.__name__)
        self.cube = cube
        self.labels = labels
        self.train_indices, self.test_indices = train_test_split(
            self._find_intersection(), test_size=test_size, random_state=seed
        )
        self.logger.info(
            "{}(Training), {}(Testing) are going to be used".format(
                self.train_indices.shape[0], self.test_indices.shape[0]
            )
        )
        self.seed = seed

    def _find_intersection(self):
        """find the galaxies in the dataset with desired values"""
        redshifts = self.labels[REDSHIFT_KEY]
        dered_petro_mag = self.labels[DEREDENED_PETRO_KEY]

        (idxes_redshifts,) = (redshifts <= MAX_REDSHIFT_VALUES).nonzero()
        (idxes_dered,) = (dered_petro_mag <= MAX_DERED_PETRO_MAG).nonzero()
        intersection = np.intersect1d(
            idxes_redshifts, idxes_dered, return_indices=False
        )

        self.logger.info(
            "There are {} galaxies with redshift ".format(intersection.shape[0])
            + "values between (0, {}] and ".format(MAX_REDSHIFT_VALUES)
            + "dered_petro_mag between (0, {}].".format(MAX_DERED_PETRO_MAG)
        )

        return intersection

    def get_k_fold_sequences(self, num_folds=5, **kwargs):
        """Return `k-folds` RedShiftDataCubeSequences from the dataset."""
        folder = KFold(n_splits=num_folds, random_state=self.seed, shuffle=True)
        X = self.train_indices

        folds = {}
        for fold_no, (train, test) in enumerate(folder.split(X), start=1):
            folds["Fold_{}".format(fold_no)] = {
                "train": RedShiftDataCubeSequence(
                    self.cube, self.labels, train, **kwargs
                ),
                "valid": RedShiftDataCubeSequence(
                    self.cube, self.labels, test, **kwargs
                ),
            }

        return folds

    def __repr__(self):
        return "<{} cube: {}, shape: {}>".format(
            self.__class__.__name__, self.cube.shape, self.labels.shape
        )


class RedShiftDataCubeSequence(Sequence):
    """The datacube sequence for redshift

    Parameters
    ----------
    labels : np.mmap,
        The mmap array for labels on the dataset
    cube : np.mmap
        The mmap array for cube on the dataset
    idxes : np.ndarray, optional, default=None
        The indexes of interest for this sequence
    batch_size : int, default=128
        The batch size
    flip_prob : float, default=0.2
        The probability of flipping (augmentation)
    rotate_prob : float, default=0.2
        The probability of 90 degree rotation (augmentation)
    bins_range: tuple, default=(0, 0.4)
        The range in which the redshift values fall
    num_bins : int, default=180
        The number of bins to divide redshift values to
    logger: logging.logger instance
        The logger
    """

    def __init__(
        self,
        cube,
        labels,
        idxes=None,
        batch_size=128,
        num_bins=180,
        bins_range=(0, 0.4),
        flip_prob=0.2,
        rotate_prob=0.2,
        logger=None,
    ):
        self.cube = cube
        self.labels = labels
        self.batch_size = batch_size
        self.bins = self._get_bins(num_bins, bins_range)

        if logger is None:
            logger = get_logger(self.__class__.__name__)
        self.logger = logger

        if idxes is None:
            idxes = np.array(range(cube.shape[0]))
        self.idxes = idxes

        self.flip_probability = flip_prob
        self.rotate_probability = rotate_prob

    def _to_categorical(self, values):
        """Convert to categorical"""
        assert np.all(values <= self.bins[-1])
        return np.digitize(values, bins=self.bins, right=False)

    def _get_batch_indices(self, batch_no):
        """Get indices for a particular batch"""
        if batch_no >= self.__len__():
            raise IndexError(
                "Batch number should be less than or equal to {}".format(len(self))
            )
        else:
            return self.idxes[
                batch_no * self.batch_size : batch_no * self.batch_size
                + self.batch_size
            ]

    def _get_batch(self, index, by_category=False, augment=False):
        batch_indices = self.get_batch_indices(index)
        batch_cube = self.cube[batch_indices]
        if augment:
            flips = np.where(
                np.random.rand(*batch_indices.shape) < self.flip_probability,
                batch_indices,
                -1,
            )
            rots = np.where(
                np.random.rand(*batch_indices.shape) < self.rotate_probability,
                batch_indices,
                -1,
            )
            flips = flips[flips >= 0]
            rots = rots[rots >= 0]
            batch_cube[flips] = np.flip(batch_cube[flips])
            for rot in rots:
                batch_cube[rot] = np.rot90(batch_cube[rot], k=np.random.randint(1, 5))

        z_truth = self.labels[REDSHIFT_KEY][batch_indices]
        if by_category:
            z_truth = self._to_categorical(z_truth)

        ebv = np.expand_dims(self.labels[EBV_KEY][batch_indices], -1)
        return (batch_cube, ebv), z_truth

    def on_epoch_end(self):
        self._reshuffle_indexes()

    def _reshuffle_indexes(self):
        np.random.shuffle(self.idxes)

    def __len__(self):
        return np.ceil(self.idxes.shape[0] / self.batch_size).astype(np.int)

    def __getitem__(self, index):
        return self._get_batch(index, by_category=True, augment=True)

    def __repr__(self):
        return "<{} num batches: {} batch size: {}>".format(
            self.__class__.__name__, len(self), self.batch_size
        )

    @staticmethod
    def _get_bins(num_bins, bins_range):
        bins = np.linspace(*bins_range, num_bins + 1)
        return bins[1:]
