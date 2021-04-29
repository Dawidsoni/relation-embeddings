import functools
import tensorflow as tf
import gin.tf


class SelfAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, heads_count: int, attention_head_dimension: int, dropout_rate: float = 0.0):
        super(SelfAttentionLayer, self).__init__()
        self.attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=heads_count,
            key_dim=attention_head_dimension,
            dropout=dropout_rate,
        )

    def call(self, inputs, mask=None, training=None):
        return self.attention_layer(query=inputs, value=inputs, attention_mask=mask, training=training)


class PointwiseFeedforwardLayer(tf.keras.layers.Layer):

    def __init__(self, hidden_layer_dimension: int):
        super(PointwiseFeedforwardLayer, self).__init__()
        self.hidden_layer_dimension = hidden_layer_dimension
        self.dense_layer1 = None
        self.dense_layer2 = None

    def build(self, input_shape):
        self.dense_layer1 = tf.keras.layers.Dense(
            self.hidden_layer_dimension, activation="relu"
        )
        self.dense_layer2 = tf.keras.layers.Dense(units=input_shape[-1])

    def call(self, inputs, training=None):
        outputs = self.dense_layer1(inputs, training=training)
        return self.dense_layer2(outputs, training=training)


class TransformerEncoderLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        attention_heads_count: int,
        attention_head_dimension: int,
        pointwise_hidden_layer_dimension: int,
        dropout_rate: float = 0.0,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.attention_layer = SelfAttentionLayer(
            heads_count=attention_heads_count,
            attention_head_dimension=attention_head_dimension,
            dropout_rate=dropout_rate,
        )
        self.pointwise_layer = PointwiseFeedforwardLayer(
            hidden_layer_dimension=pointwise_hidden_layer_dimension,
        )
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_layer1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_layer2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        attention_outputs = self.attention_layer(inputs, training=training)
        attention_outputs = self.dropout_layer1(attention_outputs, training=training)
        attention_outputs = self.layer_norm1(inputs + attention_outputs)
        pointwise_outputs = self.pointwise_layer(attention_outputs)
        pointwise_outputs = self.dropout_layer2(pointwise_outputs, training=training)
        return self.layer_norm2(attention_outputs + pointwise_outputs)


@gin.configurable
class StackedTransformerEncodersLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        layers_count: int = gin.REQUIRED,
        attention_heads_count: int = gin.REQUIRED,
        attention_head_dimension: int = gin.REQUIRED,
        pointwise_hidden_layer_dimension: int = gin.REQUIRED,
        dropout_rate: float = gin.REQUIRED,
        share_encoder_parameters: bool = gin.REQUIRED,
    ):
        super(StackedTransformerEncodersLayer, self).__init__()
        encoder_layer_initializer = functools.partial(
            TransformerEncoderLayer,
            attention_heads_count=attention_heads_count,
            attention_head_dimension=attention_head_dimension,
            pointwise_hidden_layer_dimension=pointwise_hidden_layer_dimension,
            dropout_rate=dropout_rate,
        )
        if share_encoder_parameters:
            shared_encoder_layer = encoder_layer_initializer()
            self.sublayers = [shared_encoder_layer for _ in range(layers_count)]
        else:
            self.sublayers = [encoder_layer_initializer() for _ in range(layers_count)]

    def call(self, inputs, training=None):
        outputs = inputs
        for sublayer in self.sublayers:
            outputs = sublayer(outputs, training=training)
        return outputs
