import numpy as np

from deepforge_redshift.data_loaders import DataSetSampler


class TestDataSetSampler:
    def execute(self, labels, cube):
        sampler = DataSetSampler(labels=labels, cube=cube)

        sampler2 = DataSetSampler(labels=labels, cube=cube)

        assert np.allclose(sampler.test_indices, sampler2.test_indices)
        assert np.allclose(sampler.train_indices, sampler2.train_indices)
        assert len(sampler.get_k_fold_sequences()) == len(
            sampler.get_k_fold_sequences()
        )
