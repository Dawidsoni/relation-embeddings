import dataclasses
from dataclasses import dataclass
import tensorflow as tf
import gin.tf

from layers.embeddings_layers import EmbeddingsConfig
from models.conv_base_model import KnowledgeCompletionModel


@gin.configurable
@dataclass
class ConvEModelConfig(object):
    embeddings_width: int
    input_dropout_rate: float
    conv_layer_filters: int
    conv_layer_kernel_size: int
    conv_dropout_rate: float
    hidden_dropout_rate: float


@gin.configurable(blacklist=['embeddings_config'])
class ConvEModel(KnowledgeCompletionModel):

    def __init__(self, embeddings_config: EmbeddingsConfig, model_config: ConvEModelConfig):
        super().__init__(embeddings_config)
        self.model_config = model_config
        if self.embeddings_layer.config.embeddings_dimension % self.model_config.embeddings_width != 0:
            raise ValueError("Embeddings dimension must be divisible by embeddings width")
        self.input_batch_norm_layer = tf.keras.layers.BatchNormalization(axis=1)
        self.input_dropout_layer = tf.keras.layers.Dropout(rate=self.model_config.input_dropout_rate)
        self.conv_layer = tf.keras.layers.Conv2D(
            filters=self.model_config.conv_layer_filters,
            kernel_size=self.model_config.conv_layer_kernel_size,
        )
        self.conv_batch_norm_layer = tf.keras.layers.BatchNormalization(axis=1)
        self.conv_dropout_layer = tf.keras.layers.SpatialDropout2D(rate=self.model_config.conv_dropout_rate)
        self.hidden_layer = tf.keras.layers.Dense(units=self.embeddings_layer.config.embeddings_dimension)
        self.hidden_dropout_layer = tf.keras.layers.Dropout(rate=self.model_config.hidden_dropout_rate)
        self.hidden_batch_norm_layer = tf.keras.layers.BatchNormalization(axis=1)
        projection_shape = (self.embeddings_layer.config.entities_count, )
        self.projection_bias = tf.Variable(shape=projection_shape, initial_value=tf.zeros(shape=projection_shape))

    def _embeddings_slice_as_image(self, embeddings, indexes):
        embeddings = tf.gather(embeddings, indexes, axis=1, batch_dims=1)
        height_of_embeddings = self.embeddings_layer.config.embeddings_dimension // self.model_config.embeddings_width
        return tf.reshape(
            embeddings,
            shape=(-1, 1, height_of_embeddings, self.model_config.embeddings_width)
        )

    def get_config(self):
        return {
            **dataclasses.asdict(self.embeddings_config),
            **dataclasses.asdict(self.model_config),
        }

    def call(self, inputs, training=None, **kwargs):
        embeddings = self.embeddings_layer(inputs)
        entity_indexes = inputs["true_entity_index"]
        input_entity_embeddings = self._embeddings_slice_as_image(embeddings, entity_indexes)
        relation_embeddings = self._embeddings_slice_as_image(embeddings, tf.ones_like(entity_indexes, dtype=tf.int32))
        outputs = tf.concat([input_entity_embeddings, relation_embeddings], axis=2)
        outputs = self.input_batch_norm_layer(outputs, training=training)
        outputs = self.input_dropout_layer(outputs, training=training)
        outputs = tf.transpose(outputs, perm=(0, 2, 3, 1))
        outputs = self.conv_layer(outputs, training=training)
        outputs = tf.transpose(outputs, perm=(0, 3, 1, 2))
        outputs = self.conv_batch_norm_layer(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.conv_dropout_layer(outputs, training=training)
        outputs = tf.reshape(outputs, shape=(tf.shape(outputs)[0], -1))
        outputs = self.hidden_layer(outputs, training=training)
        outputs = self.hidden_dropout_layer(outputs, training=training)
        outputs = self.hidden_batch_norm_layer(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = tf.linalg.matmul(outputs, self.embeddings_layer.entity_embeddings, transpose_b=True)
        outputs += self.projection_bias
        return tf.nn.sigmoid(outputs)
