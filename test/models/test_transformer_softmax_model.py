import dataclasses
import tensorflow as tf
import gin.tf
import numpy as np

from layers.embeddings_layers import EmbeddingsConfig
from models.transformer_softmax_model import TransformerSoftmaxModel, TransformerSoftmaxModelConfig
from optimization.datasets import MaskedEntityOfEdgeDataset, DatasetType


class TestTransformerSoftmaxModel(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def setUp(self):
        gin.clear_config()
        gin.parse_config("""
            StackedTransformerEncodersLayer.layers_count = 3
            StackedTransformerEncodersLayer.attention_heads_count = 4
            StackedTransformerEncodersLayer.attention_head_dimension = 5
            StackedTransformerEncodersLayer.pointwise_hidden_layer_dimension = 4
            StackedTransformerEncodersLayer.dropout_rate = 0.5
            StackedTransformerEncodersLayer.share_encoder_parameters = False
            StackedTransformerEncodersLayer.share_encoder_parameters = False
            StackedTransformerEncodersLayer.encoder_layer_type = %TransformerEncoderLayerType.PRE_LAYER_NORM
        """)
        dataset = MaskedEntityOfEdgeDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=5
        )
        self.model_inputs = next(iter(dataset.samples))
        self.embeddings_config = EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=6, use_special_token_embeddings=True,
        )
        self.default_model_config = TransformerSoftmaxModelConfig(
            use_pre_normalization=True, pre_dropout_rate=0.5, use_sigmoid_as_output_layer=False,
        )

    def test_model_architecture(self):
        model = TransformerSoftmaxModel(self.embeddings_config, self.default_model_config)
        outputs = model(self.model_inputs)
        self.assertAllEqual((5, 3), outputs.shape)
        self.assertEqual(747, model.count_params())

    def test_pre_normalization_disabled(self):
        model_config = dataclasses.replace(self.default_model_config, use_pre_normalization=False)
        model = TransformerSoftmaxModel(self.embeddings_config, model_config)
        outputs = model(self.model_inputs)
        self.assertAllEqual((5, 3), outputs.shape)
        self.assertEqual(735, model.count_params())

    def test_softmax_outputs(self):
        model = TransformerSoftmaxModel(self.embeddings_config, self.default_model_config)
        outputs = model(self.model_inputs)
        self.assertAllClose(tf.ones(shape=(tf.shape(outputs)[0], )), tf.reduce_sum(outputs, axis=1))

    def test_sigmoid_outputs(self):
        np.random.seed(2)
        model_config = dataclasses.replace(self.default_model_config, use_sigmoid_as_output_layer=True)
        model = TransformerSoftmaxModel(self.embeddings_config, model_config)
        outputs = model(self.model_inputs)
        self.assertNotAllClose(tf.ones(shape=(tf.shape(outputs)[0], )), tf.reduce_sum(outputs, axis=1))


if __name__ == '__main__':
    tf.test.main()
