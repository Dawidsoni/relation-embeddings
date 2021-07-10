import numpy as np
from unittest import mock
from unittest.mock import MagicMock
import tensorflow as tf
import gin.tf

from layers.embeddings_layers import ObjectType
from models.conv_base_model import EmbeddingsConfig, ConvModelConfig
from models.conve_model import ConvEModel, ConvEModelConfig
from optimization.datasets import SamplingEdgeDataset, DatasetType, MaskedEntityOfEdgeDataset
from optimization.model_trainers import SamplingModelTrainer, SoftmaxModelTrainer
from optimization.loss_objects import NormLossObject, CrossEntropyLossObject
from models.transe_model import TranseModel


class TestModelTrainers(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def setUp(self):
        gin.clear_config()
        gin.parse_config("""
            LossObject.regularization_strength = 0.1
        """)
        self.init_entity_embeddings = tf.ones(shape=(3, 4))
        self.init_relation_embeddings = 2 * tf.ones(shape=(2, 4))

    def test_sampling_model_trainer(self):
        tf.random.set_seed(1)

        embeddings_config = EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4,
            pretrained_entity_embeddings=self.init_entity_embeddings,
            pretrained_relation_embeddings=self.init_relation_embeddings,
        )
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        loss_object = NormLossObject(order=2, margin=1.0)
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=1, decay_rate=0.5
        )
        transe_model = TranseModel(embeddings_config, model_config)
        model_trainer = SamplingModelTrainer(transe_model, loss_object, learning_rate_schedule)
        dataset = SamplingEdgeDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=2
        )
        samples_iterator = iter(dataset.samples)
        model_trainer.train_step(training_samples=next(samples_iterator), training_step=1)
        model_trainer.train_step(training_samples=next(samples_iterator), training_step=2)
        self.assertEqual(2, model_trainer.optimizer.iterations)
        expected_kernel = np.array([[[[1.0]], [[1.0]], [[-1.0]]]])
        self.assertAllEqual(expected_kernel, transe_model.conv_layers[0].kernel.numpy())
        embeddings_layer = transe_model.embeddings_layer
        self.assertGreater(np.sum(self.init_entity_embeddings != embeddings_layer.entity_embeddings), 0)
        self.assertGreater(np.sum(self.init_relation_embeddings != embeddings_layer.relation_embeddings), 0)

    @mock.patch.object(tf, 'GradientTape')
    @mock.patch.object(tf.keras.optimizers, 'Adam')
    def test_sampling_trainer_multiple_negatives(self, unused_adam_mock, unused_gradient_tape_mock):
        np.random.seed(2)
        dataset = SamplingEdgeDataset(
            dataset_type=DatasetType.TRAINING,
            data_directory=self.DATASET_PATH,
            batch_size=4,
            negatives_per_positive=2,
        )
        get_losses_of_pairs_mock = MagicMock(side_effect=[tf.constant([2.0, 3.0]), tf.constant([1.0, 4.0])])
        get_regularization_loss_mock = MagicMock(side_effect=[5.0])
        model_mock = MagicMock()
        loss_object_mock = MagicMock(
            get_losses_of_pairs=get_losses_of_pairs_mock, get_regularization_loss=get_regularization_loss_mock
        )
        model_trainer = SamplingModelTrainer(
            model=model_mock, loss_object=loss_object_mock, learning_rate_schedule=MagicMock()
        )
        training_samples = next(iter(dataset.samples))
        loss_value = model_trainer.train_step(training_samples, training_step=1)
        self.assertEqual(8.0, loss_value)
        edge_object_types = np.broadcast_to(
            [ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value],
            shape=(4, 3)
        ).tolist()
        print(model_mock.call_args_list[0][0][0]["object_ids"])
        self.assertAllEqual([[0, 0, 1], [1, 1, 2], [0, 0, 1], [1, 1, 2]],
                            model_mock.call_args_list[0][0][0]["object_ids"])
        self.assertAllEqual(edge_object_types, model_mock.call_args_list[0][0][0]["object_types"])
        self.assertAllEqual([[0, 0, 0], [0, 1, 2], [2, 0, 1], [1, 1, 0]],
                            model_mock.call_args_list[1][0][0]["object_ids"])
        self.assertAllEqual(edge_object_types, model_mock.call_args_list[1][0][0]["object_types"])
        self.assertAllEqual([[0, 0, 2], [2, 1, 2], [1, 0, 1], [1, 1, 1]],
                            model_mock.call_args_list[2][0][0]["object_ids"])
        self.assertAllEqual(edge_object_types, model_mock.call_args_list[2][0][0]["object_types"])

    def test_softmax_model_trainer(self):
        dataset = MaskedEntityOfEdgeDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=5
        )
        embeddings_config = EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4, use_special_token_embeddings=True,
            pretrained_entity_embeddings=self.init_entity_embeddings,
            pretrained_relation_embeddings=self.init_relation_embeddings,
        )
        model_config = ConvEModelConfig(
            embeddings_width=2, input_dropout_rate=0.5, conv_layer_filters=32, conv_layer_kernel_size=2,
            conv_dropout_rate=0.5, hidden_dropout_rate=0.5,
        )
        model = ConvEModel(embeddings_config, model_config)
        loss_object = CrossEntropyLossObject(label_smoothing=0.1)
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=1, decay_rate=0.5
        )
        trainer = SoftmaxModelTrainer(model, loss_object, learning_rate_schedule)
        trainer.train_step(training_samples=next(iter(dataset.samples)), training_step=1)
        self.assertGreater(np.sum(self.init_entity_embeddings != model.embeddings_layer.entity_embeddings), 0)
        self.assertGreater(np.sum(self.init_relation_embeddings != model.embeddings_layer.relation_embeddings), 0)


if __name__ == '__main__':
    tf.test.main()
