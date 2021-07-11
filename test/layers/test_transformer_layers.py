import tensorflow as tf

from layers.transformer_layers import SelfAttentionLayer, PointwiseFeedforwardLayer, TransformerEncoderLayer, \
    StackedTransformerEncodersLayer, PreNormalizationTransformerEncoderLayer, PostNormalizationTransformerEncoderLayer, \
    TransformerEncoderLayerType


class TestTransformerLayers(tf.test.TestCase):

    def test_self_attention_layer(self):
        layer = SelfAttentionLayer(heads_count=4, attention_head_dimension=512, dropout_rate=0.5)
        input_embeddings = tf.ones(shape=(32, 10, 512))
        output_embeddings = layer(input_embeddings)
        self.assertEqual((32, 10, 512), output_embeddings.shape)

    def test_pointwise_feedforward_layer(self):
        layer = PointwiseFeedforwardLayer(hidden_layer_dimension=512)
        input_embeddings = tf.ones(shape=(32, 10, 512))
        output_embeddings = layer(input_embeddings)
        self.assertEqual((32, 10, 512), output_embeddings.shape)

    def test_pre_normalization_transformer_encoder_layer(self):
        layer = PreNormalizationTransformerEncoderLayer(
            attention_heads_count=4,
            attention_head_dimension=512,
            pointwise_hidden_layer_dimension=512,
            dropout_rate=0.5,
        )
        input_embeddings = tf.ones(shape=(32, 10, 512))
        output_embeddings = layer(input_embeddings)
        self.assertEqual((32, 10, 512), output_embeddings.shape)

    def test_post_normalization_transformer_encoder_layer(self):
        layer = PostNormalizationTransformerEncoderLayer(
            attention_heads_count=4,
            attention_head_dimension=512,
            pointwise_hidden_layer_dimension=512,
            dropout_rate=0.5,
        )
        input_embeddings = tf.ones(shape=(32, 10, 512))
        output_embeddings = layer(input_embeddings)
        self.assertEqual((32, 10, 512), output_embeddings.shape)

    def test_stacked_transformer_encoder_layers(self):
        layer = StackedTransformerEncodersLayer(
            layers_count=12,
            attention_heads_count=8,
            attention_head_dimension=512,
            pointwise_hidden_layer_dimension=2048,
            dropout_rate=0.5,
            share_encoder_parameters=False,
            encoder_layer_type=TransformerEncoderLayerType.POST_LAYER_NORM,
        )
        input_embeddings = tf.ones(shape=(32, 10, 512))
        output_embeddings = layer(input_embeddings)
        self.assertEqual((32, 10, 512), output_embeddings.shape)

    def test_stacked_transformer_encoder_params(self):
        layer = StackedTransformerEncodersLayer(
            layers_count=12,
            attention_heads_count=8,
            attention_head_dimension=512,
            pointwise_hidden_layer_dimension=2048,
            dropout_rate=0.5,
            share_encoder_parameters=False,
            encoder_layer_type=TransformerEncoderLayerType.POST_LAYER_NORM,
        )
        layer(tf.ones(shape=(32, 10, 512)))
        self.assertEqual(126_038_016, layer.count_params())

    def test_stacked_transformer_encoder_shared_layer_params(self):
        layer = StackedTransformerEncodersLayer(
            layers_count=12,
            attention_heads_count=8,
            attention_head_dimension=512,
            pointwise_hidden_layer_dimension=2048,
            dropout_rate=0.5,
            share_encoder_parameters=True,
            encoder_layer_type=TransformerEncoderLayerType.POST_LAYER_NORM,
        )
        layer(tf.ones(shape=(32, 10, 512)))
        self.assertEqual(10_503_168, layer.count_params())

    def test_stacked_transformer_encoder_pre_layer_norm(self):
        layer = StackedTransformerEncodersLayer(
            layers_count=2,
            attention_heads_count=8,
            attention_head_dimension=512,
            pointwise_hidden_layer_dimension=2048,
            dropout_rate=0.5,
            share_encoder_parameters=False,
            encoder_layer_type=TransformerEncoderLayerType.PRE_LAYER_NORM,
        )
        self.assertLen(layer.sublayers, 3)
        self.assertIsInstance(layer.sublayers[0], PreNormalizationTransformerEncoderLayer)
        self.assertIsInstance(layer.sublayers[1], PreNormalizationTransformerEncoderLayer)
        self.assertIsInstance(layer.sublayers[2], tf.keras.layers.LayerNormalization)

    def test_stacked_transformer_encoder_post_layer_norm(self):
        layer = StackedTransformerEncodersLayer(
            layers_count=2,
            attention_heads_count=8,
            attention_head_dimension=512,
            pointwise_hidden_layer_dimension=2048,
            dropout_rate=0.5,
            share_encoder_parameters=False,
            encoder_layer_type=TransformerEncoderLayerType.POST_LAYER_NORM,
        )
        self.assertLen(layer.sublayers, 2)
        self.assertIsInstance(layer.sublayers[0], PostNormalizationTransformerEncoderLayer)
        self.assertIsInstance(layer.sublayers[1], PostNormalizationTransformerEncoderLayer)
