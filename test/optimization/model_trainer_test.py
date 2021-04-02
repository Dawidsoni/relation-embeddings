import numpy as np
import tensorflow as tf

from models.conv_base_model import EmbeddingsConfig, ConvModelConfig
from data_handlers.dataset import Dataset
from optimization.model_trainers import ModelTrainer
from models.transe_model import TranseModel
from data_handlers.losses import LossObject, OptimizedMetric


class TestModelTrainer(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def test_train_model(self):
        tf.random.set_seed(1)
        pretrained_entity_embeddings = tf.ones(shape=(3, 4))
        pretrained_relation_embeddings = 2 * tf.ones(shape=(2, 4))
        embeddings_config = EmbeddingsConfig(
            entities_count=3, relations_count=2, pretrained_entity_embeddings=pretrained_entity_embeddings,
            pretrained_relation_embeddings=pretrained_relation_embeddings
        )
        model_config = ConvModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False)
        loss_object = LossObject(
            OptimizedMetric.NORM, regularization_strength=0.1, norm_metric_order=2, norm_metric_margin=1.0
        )
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=1, decay_rate=0.5
        )
        transe_model = TranseModel(embeddings_config, model_config)
        model_trainer = ModelTrainer(transe_model, loss_object, learning_rate_schedule)
        dataset = Dataset(graph_edges_filename='train.txt', data_directory=self.DATASET_PATH, batch_size=2,
                          repeat_samples=True)
        pairs_of_samples_iterator = iter(dataset.pairs_of_samples)
        positive_inputs1, negative_inputs1 = next(pairs_of_samples_iterator)
        positive_inputs2, negative_inputs2 = next(pairs_of_samples_iterator)
        model_trainer.train_step(positive_inputs1, negative_inputs1, training_step=1)
        model_trainer.train_step(positive_inputs2, negative_inputs2, training_step=2)
        self.assertEqual(2, model_trainer.optimizer.iterations)
        expected_kernel = np.array([[[[1.0]], [[1.0]], [[-1.0]]]])
        self.assertAllEqual(expected_kernel, transe_model.conv_layers[0].kernel.numpy())
        self.assertGreater(np.sum(pretrained_entity_embeddings != transe_model.entity_embeddings), 0)
        self.assertGreater(np.sum(pretrained_relation_embeddings != transe_model.relation_embeddings), 0)


if __name__ == '__main__':
    tf.test.main()
