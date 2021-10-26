import numpy as np
import tensorflow as tf
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
    fraction: float, default=0.2
        The Fraction of the dataset to train on
    number_of_epochs: int
        The number of epochs to train for
    """

    def __init__(self, model, fraction=0.2, number_of_epochs=30):
        self.fraction = fraction
        self.number_of_epochs = number_of_epochs
        self.model = model
        self.logger = get_logger(self.__class__.__name__)

    def execute(self, cube, labels):
        """Execute the train operation."""
        self._compile_model()
        sampler = DataSetSampler(cube=cube, labels=labels)

        init_weights = self.model.get_weights()
        folds = sampler.get_k_fold_sequences(num_folds=5, fraction=self.fraction)
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
            fold_weights[fold_name] = self.model.get_weights()

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


class ResiudalsEvaluator:
    def __init__(self, model, rs_max_val=0.4, rs_num_bins=180):
        self.model = model
        self.logger = get_logger(self.__class__.__name__)
        self.rs_max_val = rs_max_val
        self.rs_num_bins = rs_num_bins

    def execute(self, fold_weights, cube, labels):
        # Execute your operation here!

        self._compile_model()
        sampler = DataSetSampler(
            cube=cube,
            labels=labels
        )
        test_sequence = sampler.get_test_sequence()
        # print(test_sequence[0].shape)

        fold_predictions = {}
        for weight in fold_weights:
            self.logger.info(f'Evaluating residuals for {self.model.name}; Fold {weight}')
            self.model.set_weights(fold_weights[weight])
            predictions = []
            truths = []
            for sequence, z_truth in test_sequence:
                preds_softmax = self.model.predict_on_batch(sequence)
                preds = self.get_prediction(preds_softmax).numpy()
                predictions.extend(preds)
                truths.extend(z_truth)
                assert preds.shape == z_truth.shape
            fold_predictions[weight] = {
                'predictions': predictions,
                'truths': truths
            }

        return fold_predictions

    def get_prediction(self, y_pred):
        step = self.rs_max_val / self.rs_num_bins
        bins = np.arange(0, self.rs_max_val, step) + (step / 2)
        y_prediction = tf.reduce_sum(tf.multiply(y_pred, bins), axis=1)
        return y_prediction

    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(lr=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )

        self.logger.info(
            "Successfully compiled the model with `sparse_categorical_crossentropy` loss"
        )

class EnsembleMetrics:
    def execute(self, res1, res2, res3, res4, res5, res6):
        pass

    # def


if __name__ == "__main__":
    import numpy as np
    import pickle
    from deepforge_redshift.models.inception_seed32 import model as model_seed_32
    from deepforge_redshift.models.inception_seed42 import model as model_seed_42
    from deepforge_redshift.models.inception_seed52 import model as model_seed_52
    from deepforge_redshift.models.inception_seed62 import model as model_seed_62
    from deepforge_redshift.models.inception_seed72 import model as model_seed_72
    from deepforge_redshift.models.inception_seed82 import model as model_seed_82

    cube = np.load('/home/umesh/worker-cache/s3/astrodata/cube.npy', mmap_mode='r')
    labels = np.load('/home/umesh/worker-cache/s3/astrodata/labels.npy', mmap_mode='r')

    models = [
        model_seed_32,
        model_seed_42,
        model_seed_52,
        model_seed_62,
        model_seed_72,
        model_seed_82
    ]
    weights = [
        'f_0.2_seed_32',
        'f_0.2_seed_42',
        'f_0.2_seed_52',
        'f_0.2_seed_62',
        'f_0.2_seed_72',
        'f_0.2_seed_82'
    ]
    all_residuals = []
    for model, weights_name in zip(models, weights):
        with open(f'./deepforge_redshift/{weights_name}', 'rb') as pcl:
            data_dict = pickle.load(pcl)
            eval1 = ResiudalsEvaluator(model=model)
            all_residuals.append(eval1.execute(fold_weights=data_dict, labels=labels, cube=cube))

    with open('residuals.pcl', 'wb') as pickle_file:
        pickle.dump(all_residuals, pickle_file)
        print(f'Successfully saved residuals as {pickle_file.name}')