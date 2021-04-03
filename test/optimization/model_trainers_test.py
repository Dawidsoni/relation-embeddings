import numpy as np
import tensorflow as tf

from models.conv_base_model import EmbeddingsConfig, ConvModelConfig
from optimization.datasets import SamplingDataset, DatasetType
from optimization.model_trainers import SamplingModelTrainer
from optimization.loss_objects import NormLossObject
from models.transe_model import TranseModel


class TestModelTrainer(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def test_sampling_model_trainer(self):
        tf.random.set_seed(1)
        pretrained_entity_embeddings = tf.ones(shape=(3, 4))
        pretrained_relation_embeddings = 2 * tf.ones(shape=(2, 4))
        embeddings_config = EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4,
            pretrained_entity_embeddings=pretrained_entity_embeddings,
            pretrained_relation_embeddings=pretrained_relation_embeddings
        )
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        loss_object = NormLossObject(
            regularization_strength=0.1, metric_order=2, metric_margin=1.0
        )
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=1, decay_rate=0.5
        )
        transe_model = TranseModel(embeddings_config, model_config)
        model_trainer = SamplingModelTrainer(transe_model, loss_object, learning_rate_schedule)
        dataset = SamplingDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=2, repeat_samples=True
        )
        pairs_of_samples_iterator = iter(dataset.pairs_of_samples)
        model_trainer.train_step(training_samples=next(pairs_of_samples_iterator), training_step=1)
        model_trainer.train_step(training_samples=next(pairs_of_samples_iterator), training_step=2)
        self.assertEqual(2, model_trainer.optimizer.iterations)
        expected_kernel = np.array([[[[1.0]], [[1.0]], [[-1.0]]]])
        self.assertAllEqual(expected_kernel, transe_model.conv_layers[0].kernel.numpy())
        self.assertGreater(np.sum(pretrained_entity_embeddings != transe_model.entity_embeddings), 0)
        self.assertGreater(np.sum(pretrained_relation_embeddings != transe_model.relation_embeddings), 0)


if __name__ == '__main__':
    tf.test.main()
