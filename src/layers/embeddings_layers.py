from dataclasses import dataclass
from typing import Optional
import numpy as np
import tensorflow as tf
import gin.tf


@gin.configurable
@dataclass
class EmbeddingsConfig(object):
    entities_count: int
    relations_count: int
    embeddings_dimension: int
    trainable_embeddings: bool = True
    pretrained_entity_embeddings: Optional[np.ndarray] = None
    pretrained_relations_embeddings: Optional[np.ndarray] = None
    pretrained_mask_embedding: Optional[np.ndarray] = None
    use_position_embeddings: bool = False
    position_embeddings_max_inputs_length: int = 3
    use_fourier_series_in_position_embeddings: bool = False
    position_embeddings_trainable = False


class PositionEmbeddingsLayer(tf.keras.layers.Layer):

    def __init__(self, max_inputs_length: int, use_fourier_series: bool, trainable: bool):
        super(PositionEmbeddingsLayer, self).__init__()
        self.max_inputs_length = max_inputs_length
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
        if self.use_fourier_series:
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
        self.position_embeddings_layer = PositionEmbeddingsLayer(
            max_inputs_length=config.position_embeddings_max_inputs_length,
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
            self._get_initial_embedding_values(relations_shape, self.config.pretrained_relations_embeddings),
            name='relation_embeddings',
            trainable=self.config.trainable_embeddings
        )

    def call(self, inputs, **kwargs):
        head_entity_ids, relation_ids, tail_entity_ids = tf.unstack(inputs, axis=1)
        head_entity_embeddings = tf.gather(self.entity_embeddings, head_entity_ids)
        relation_embeddings = tf.gather(self.relation_embeddings, relation_ids)
        tail_entity_embeddings = tf.gather(self.entity_embeddings, tail_entity_ids)
        return head_entity_embeddings, relation_embeddings, tail_entity_embeddings
