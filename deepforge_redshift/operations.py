import numpy as np
from tensorflow.keras.optimizers import Adam

from deepforge_redshift.data_loaders import DataSetSampler
from deepforge_redshift.utils import get_logger


class TestDataSetSampler:
    def execute(self, labels, cube):
        sampler = DataSetSampler(labels=labels, cube=cube)

        sampler2 = DataSetSampler(labels=labels, cube=cube)

        assert np.allclose(sampler.test_indices, sampler2.test_indices)
        assert np.allclose(sampler.train_indices, sampler2.train_indices)
        assert len(sampler.get_k_fold_sequences()) == len(
            sampler.get_k_fold_sequences()
        )


class KFoldsTraining:
    """Perform a K-Folds training for redshift using a particular model.

    Parameters
    ----------
    model: tensorflow.keras.Model
        The model to train it on
    number_of_epochs: int
        The number of epochs to train for
    """

    def __init__(self, model, number_of_epochs=30):
        self.number_of_epochs = number_of_epochs
        self.model = model
        self.logger = get_logger(self.__class__.__name__)

    def execute(self, cube, labels):
        """Execute the train operation."""
        self._compile_model()
        sampler = DataSetSampler(cube=cube, labels=labels)

        init_weights = self.model.get_weights()
        folds = sampler.get_k_fold_sequences(num_folds=5)
        fold_weights = {}

        for fold_name, sequence_dict in folds.items():
            self.logger.info("Currently training {}".format(fold_name))
            self.model.set_weights(init_weights)
            self.model.fit(
                sequence_dict["train"],
                validation_data=sequence_dict["valid"],
                epochs=self.number_of_epochs,
            )
            self.logger.info("The training for {} is complete".format(fold_name))
            fold_weights[fold_name] = model.get_weights()

        return fold_weights

    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(lr=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )

        self.logger.info(
            "Successfully compiled the model with `sparse_categorical_crossentropy` loss"
        )
