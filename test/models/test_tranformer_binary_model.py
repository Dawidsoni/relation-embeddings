import tensorflow as tf
import gin.tf

from layers.embeddings_layers import EmbeddingsConfig, ObjectType
from models.transformer_binary_model import TransformerBinaryModel


class TestTransformerBinaryModel(tf.test.TestCase):

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
            StackedTransformerEncodersLayer.encoder_layer_type = %TransformerEncoderLayerType.POST_LAYER_NORM
        """)
        edge_object_type = [
            ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value, ObjectType.ENTITY.value,
            ObjectType.RELATION.value
        ]
        self.inputs = {
            "object_ids": tf.constant([[0, 0, 1, 1, 0], [1, 1, 2, 0, 1]], dtype=tf.int32),
            "object_types": tf.constant([edge_object_type, edge_object_type], dtype=tf.int32),
        }

    def test_model(self):
        embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=4)
        model = TransformerBinaryModel(embeddings_config, use_only_edge_in_reduction_layer=False)
        outputs = model(self.inputs)
        self.assertAllEqual((2, 1), outputs.shape)
        self.assertEqual(449, model.count_params())

    def test_using_edge_only_in_reduction_layer(self):
        embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=4)
        model = TransformerBinaryModel(embeddings_config, use_only_edge_in_reduction_layer=True)
        outputs = model(self.inputs)
        self.assertAllEqual((2, 1), outputs.shape)
        self.assertEqual(441, model.count_params())


if __name__ == '__main__':
    tf.test.main()
