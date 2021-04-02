from dataclasses import dataclass
from typing import Optional
import numpy as np
import tensorflow as tf
import gin.tf

from optimization.datasets import ObjectType


@gin.configurable
@dataclass
class EmbeddingsConfig(object):
    entities_count: int
    relations_count: int
    embeddings_dimension: int
    trainable_embeddings: bool = True
    pretrained_entity_embeddings: Optional[np.ndarray] = None
    pretrained_relation_embeddings: Optional[np.ndarray] = None
    pretrained_position_embeddings: Optional[np.ndarray] = None
    pretrained_mask_embeddings: Optional[np.ndarray] = None
    use_position_embeddings: bool = False
    position_embeddings_max_inputs_length: int = 3
    use_fourier_series_in_position_embeddings: bool = False
    position_embeddings_trainable: bool = False


class PositionEmbeddingsLayer(tf.keras.layers.Layer):

    def __init__(
        self, max_inputs_length: int, initial_values: Optional[np.ndarray], use_fourier_series: bool, trainable: bool
    ):
        super(PositionEmbeddingsLayer, self).__init__()
        if initial_values is not None and use_fourier_series:
            raise ValueError("Cannot set initial values and use fourier series at the same time")
        self.max_inputs_length = max_inputs_length
        self.initial_values = initial_values
        self.use_fourier_series = use_fourier_series
        self.trainable = trainable
        self.position_embeddings = None

    def _get_fourier_angles(self, embeddings_dimension):
        input_positions = np.arange(self.max_inputs_length).reshape((-1, 1))
        embedding_positions = np.arange(embeddings_dimension).reshape((1, -1))
        relative_embeddings_positions = (2.0 * (embedding_positions // 2)) / embeddings_dimension
        return input_positions / np.power(10000, relative_embeddings_positions)

    def _get_fourier_positional_embeddings(self, embeddings_dimension):
        angles = self._get_fourier_angles(embeddings_dimension)
        positional_embeddings = np.zeros(angles.shape)
        positional_embeddings[:, 0::2] = np.sin(angles[:, 0::2])
        positional_embeddings[:, 1::2] = np.cos(angles[:, 1::2])
        return positional_embeddings

    def _get_initial_embeddings(self, embeddings_dimension):
        if self.initial_values is not None:
            return self.initial_values
        elif self.use_fourier_series:
            return self._get_fourier_positional_embeddings(embeddings_dimension)
        return tf.random.truncated_normal(shape=(self.max_inputs_length, embeddings_dimension))

    def build(self, input_shape):
        initial_embeddings = self._get_initial_embeddings(
            embeddings_dimension=input_shape[-1]
        )
        self.position_embeddings = tf.Variable(
            initial_embeddings,
            name='position_embeddings',
            trainable=self.trainable,
        )

    def call(self, inputs, training=None):
        inputs_length = tf.shape(inputs)[-2]
        chosen_embeddings = self.position_embeddings[:inputs_length, :]
        return tf.broadcast_to(chosen_embeddings, inputs.shape)


class EmbeddingsExtractionLayer(tf.keras.layers.Layer):

    def __init__(self, config: EmbeddingsConfig):
        super(EmbeddingsExtractionLayer, self).__init__()
        self.config = config
        self.entity_embeddings = self._create_entity_embeddings_variable()
        self.relation_embeddings = self._create_relation_embeddings_variable()
        self.mask_embeddings = self._create_mask_embeddings_variable()
        self.position_embeddings_layer = PositionEmbeddingsLayer(
            max_inputs_length=config.position_embeddings_max_inputs_length,
            initial_values=config.pretrained_position_embeddings,
            use_fourier_series=config.use_fourier_series_in_position_embeddings,
            trainable=config.position_embeddings_trainable
        )

    @staticmethod
    def _get_initial_embedding_values(shape, pretrained_embeddings=None):
        if pretrained_embeddings is None:
            return tf.random.truncated_normal(shape=shape)
        return pretrained_embeddings

    def _create_entity_embeddings_variable(self):
        entities_shape = [self.config.entities_count, self.config.embeddings_dimension]
        return tf.Variable(
            self._get_initial_embedding_values(entities_shape, self.config.pretrained_entity_embeddings),
            name='entity_embeddings',
            trainable=self.config.trainable_embeddings
        )

    def _create_relation_embeddings_variable(self):
        relations_shape = [self.config.relations_count, self.config.embeddings_dimension]
        return tf.Variable(
            self._get_initial_embedding_values(relations_shape, self.config.pretrained_relation_embeddings),
            name='relation_embeddings',
            trainable=self.config.trainable_embeddings
        )

    def _create_mask_embeddings_variable(self):
        masks_shape = [1, self.config.embeddings_dimension]
        return tf.Variable(
            self._get_initial_embedding_values(masks_shape, self.config.pretrained_mask_embeddings),
            name='mask_embeddings',
            trainable=self.config.trainable_embeddings
        )

    def _extract_object_embeddings(self, object_ids, object_types):
        merged_embeddings = tf.concat([self.entity_embeddings, self.relation_embeddings, self.mask_embeddings], axis=0)
        relation_types = tf.cast(object_types == ObjectType.RELATION.value, dtype=tf.int32)
        relation_offset = tf.shape(self.entity_embeddings)[0]
        mask_types = tf.cast(object_types == ObjectType.MASK.value, dtype=tf.int32)
        mask_offset = relation_offset + tf.shape(self.relation_embeddings)[0]
        offsets = relation_offset * relation_types + mask_offset * mask_types
        padded_object_ids = object_ids + offsets
        return tf.gather(merged_embeddings, padded_object_ids)

    def call(self, inputs, **kwargs):
        object_ids, object_types = inputs
        embeddings = self._extract_object_embeddings(object_ids, object_types)
        if self.config.use_position_embeddings:
            position_embeddings = self.position_embeddings_layer(embeddings)
            embeddings = embeddings + position_embeddings
        return embeddings
