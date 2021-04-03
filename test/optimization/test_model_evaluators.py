from unittest import mock

import tensorflow as tf

from models.conv_base_model import EmbeddingsConfig, ConvModelConfig
from optimization.datasets import SamplingDataset, DatasetType
from optimization.model_evaluators import SamplingModelEvaluator
from optimization.loss_objects import NormLossObject
from models.transe_model import TranseModel


class TestModelEvaluators(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    @mock.patch.object(tf.summary, "scalar")
    def test_sampling_model_evaluator(self, summary_scalar_patch):
        embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=4)
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        transe_model = TranseModel(embeddings_config, model_config)
        loss_object = NormLossObject(regularization_strength=0.1, order=2, margin=1.0)
        learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=1, decay_rate=0.5
        )
        dataset = SamplingDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, repeat_samples=True
        )
        model_evaluator = SamplingModelEvaluator(
            model=transe_model, loss_object=loss_object, dataset=dataset, existing_graph_edges=dataset.graph_edges,
            output_directory="logs", learning_rate_scheduler=learning_rate_scheduler, samples_per_step=2
        )
        model_evaluator.evaluation_step(step=0)
        model_evaluator.evaluation_step(step=1)
        self.assertEqual(26, summary_scalar_patch.call_count)


if __name__ == '__main__':
    tf.test.main()
