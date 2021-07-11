from unittest import mock

import tensorflow as tf
import gin.tf

from models.conv_base_model import EmbeddingsConfig, ConvModelConfig
from models.conve_model import ConvEModelConfig, ConvEModel
from optimization.datasets import SamplingEdgeDataset, DatasetType, MaskedEntityOfEdgeDataset
from optimization.model_evaluators import SamplingModelEvaluator, SoftmaxModelEvaluator
from optimization.loss_objects import NormLossObject, CrossEntropyLossObject
from models.transe_model import TranseModel


class TestModelEvaluators(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def setUp(self):
        gin.clear_config()
        gin.parse_config("""
            LossObject.regularization_strength = 0.1
        """)

    @mock.patch.object(tf.summary, "scalar")
    @mock.patch.object(tf.summary, "histogram")
    @mock.patch.object(tf.summary, "create_file_writer")
    def test_sampling_model_evaluator(self, file_writer_mock, summary_scalar_histogram, summary_scalar_patch):
        embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=4)
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        transe_model = TranseModel(embeddings_config, model_config)
        loss_object = NormLossObject(order=2, margin=1.0)
        learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=1, decay_rate=0.5
        )
        dataset = SamplingEdgeDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=5,
        )
        model_evaluator = SamplingModelEvaluator(
            model=transe_model, loss_object=loss_object, dataset=dataset, existing_graph_edges=dataset.graph_edges,
            output_directory="logs", learning_rate_scheduler=learning_rate_scheduler
        )
        model_evaluator.evaluation_step(step=0)
        model_evaluator.evaluation_step(step=1)
        model_evaluator.log_metrics(logger=mock.MagicMock())
        file_writer_mock.assert_called_once()
        self.assertEqual(4, summary_scalar_histogram.call_count)
        self.assertEqual(24, summary_scalar_patch.call_count)

    @mock.patch.object(tf.summary, "scalar")
    @mock.patch.object(tf.summary, "create_file_writer")
    def test_softmax_model_evaluator(self, file_writer_mock, summary_scalar_patch):
        embeddings_config = EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4, use_special_token_embeddings=True,
        )
        model_config = ConvEModelConfig(embeddings_width=2)
        model = ConvEModel(embeddings_config, model_config)
        loss_object = CrossEntropyLossObject(label_smoothing=0.1)
        learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=1, decay_rate=0.5
        )
        dataset = MaskedEntityOfEdgeDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=5
        )
        model_evaluator = SoftmaxModelEvaluator(
            model=model, loss_object=loss_object, dataset=dataset, existing_graph_edges=dataset.graph_edges,
            output_directory="logs", learning_rate_scheduler=learning_rate_scheduler
        )
        model_evaluator.evaluation_step(step=0)
        model_evaluator.evaluation_step(step=1)
        model_evaluator.log_metrics(logger=mock.MagicMock())
        file_writer_mock.assert_called_once()
        self.assertEqual(24, summary_scalar_patch.call_count)


if __name__ == '__main__':
    tf.test.main()
