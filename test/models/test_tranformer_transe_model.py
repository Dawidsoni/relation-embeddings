import tensorflow as tf
import gin.tf

from layers.embeddings_layers import EmbeddingsConfig, ObjectType
from models.conv_base_model import ConvModelConfig
from models.transformer_transe_model import TransformerTranseModel


class TestTransformerTranseModel(tf.test.TestCase):

    def setUp(self):
        gin.clear_config()
        gin.parse_config("""
            StackedTransformerEncodersLayer.layers_count = 3
            StackedTransformerEncodersLayer.attention_heads_count = 4
            StackedTransformerEncodersLayer.attention_head_dimension = 5
            StackedTransformerEncodersLayer.pointwise_hidden_layer_dimension = 4
            StackedTransformerEncodersLayer.dropout_rate = 0.5
            StackedTransformerEncodersLayer.share_encoder_parameters = False
        """)

    def test_model(self):
        embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=4)
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        model = TransformerTranseModel(embeddings_config, model_config)
        edge_object_type = [
            ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value, ObjectType.ENTITY.value,
            ObjectType.RELATION.value
        ]
        inputs = {
            "object_ids": tf.constant([[0, 0, 1, 1, 0], [1, 1, 2, 0, 1]], dtype=tf.int32),
            "object_types": tf.constant([edge_object_type, edge_object_type], dtype=tf.int32),
        }
        outputs = model(inputs)
        self.assertAllEqual((2, 4), outputs.shape)


if __name__ == '__main__':
    tf.test.main()
