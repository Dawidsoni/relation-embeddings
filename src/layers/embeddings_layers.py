import enum
from dataclasses import dataclass
from typing import Optional
import numpy as np
import tensorflow as tf
import gin.tf

from optimization import parameters_factory


@gin.configurable
@dataclass
class EmbeddingsConfig(object):
    entities_count: int
    relations_count: int
    embeddings_dimension: int = gin.REQUIRED
    trainable_embeddings: bool = True
    use_special_token_embeddings: bool = False
    pretrained_entity_embeddings: Optional[np.ndarray] = None
    pretrained_relation_embeddings: Optional[np.ndarray] = None
    pretrained_position_embeddings: Optional[np.ndarray] = None
    pretrained_special_token_embeddings: Optional[np.ndarray] = None
    use_position_embeddings: bool = False
    position_embeddings_max_inputs_length: int = 3
    use_fourier_series_in_position_embeddings: bool = False
    position_embeddings_trainable: bool = False
    special_tokens_count: int = 5


class ObjectType(enum.Enum):
    ENTITY = 0
    RELATION = 1
    SPECIAL_TOKEN = 2


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
        return parameters_factory.get_embeddings_initializer()(shape=(self.max_inputs_length, embeddings_dimension))

    def build(self, input_shape):
        initial_embeddings = self._get_initial_embeddings(
            embeddings_dimension=input_shape[-1]
        )
        self.position_embeddings = tf.Variable(
            initial_embeddings,
            name='position_embeddings',
            trainable=self.trainable,
            dtype=tf.float32,
        )

    def call(self, inputs, positions=None, training=None):
        if positions is None:
            inputs_length = tf.shape(inputs)[-2]
            chosen_embeddings = self.position_embeddings[:inputs_length, :]
        else:
            chosen_embeddings = tf.gather(self.position_embeddings, positions)
        return tf.broadcast_to(chosen_embeddings, tf.shape(inputs))


class EmbeddingsExtractionLayer(tf.keras.layers.Layer):

    def __init__(self, config: EmbeddingsConfig):
        super(EmbeddingsExtractionLayer, self).__init__()
        self.config = config
        self.entity_embeddings = self._create_entity_embeddings_variable()
        self.relation_embeddings = self._create_relation_embeddings_variable()
        self.special_token_embeddings = self._create_special_token_embeddings_variable()
        self.position_embeddings_layer = PositionEmbeddingsLayer(
            max_inputs_length=config.position_embeddings_max_inputs_length,
            initial_values=config.pretrained_position_embeddings,
            use_fourier_series=config.use_fourier_series_in_position_embeddings,
            trainable=config.position_embeddings_trainable
        )

    @staticmethod
    def _get_initial_embedding_values(shape, pretrained_embeddings=None):
        if pretrained_embeddings is None:
            return parameters_factory.get_embeddings_initializer()(shape=shape)
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

    def _create_special_token_embeddings_variable(self):
        if not self.config.use_special_token_embeddings:
            return tf.Variable(
                np.zeros(shape=(0, self.config.embeddings_dimension), dtype=np.float32),
                name="special_token_embeddings",
                trainable=False,
            )
        special_tokens_shape = [self.config.special_tokens_count, self.config.embeddings_dimension]
        return tf.Variable(
            self._get_initial_embedding_values(special_tokens_shape, self.config.pretrained_special_token_embeddings),
            name='special_token_embeddings',
            trainable=self.config.trainable_embeddings,
        )

    def _extract_object_embeddings(self, object_ids, object_types):
        merged_embeddings = tf.concat(
            [self.entity_embeddings, self.relation_embeddings, self.special_token_embeddings],
            axis=0
        )
        relation_types = tf.cast(object_types == ObjectType.RELATION.value, dtype=tf.int32)
        relation_offset = tf.shape(self.entity_embeddings)[0]
        special_token_types = tf.cast(object_types == ObjectType.SPECIAL_TOKEN.value, dtype=tf.int32)
        special_token_offset = relation_offset + tf.shape(self.relation_embeddings)[0]
        offsets = relation_offset * relation_types + special_token_offset * special_token_types
        padded_object_ids = object_ids + offsets
        return tf.gather(merged_embeddings, padded_object_ids)

    def _maybe_add_position_embeddings(self, inputs, embeddings):
        if not self.config.use_position_embeddings:
            return embeddings
        if "positions" in inputs:
            return embeddings + self.position_embeddings_layer(embeddings, positions=inputs["positions"])
        return embeddings + self.position_embeddings_layer(embeddings)

    def call(self, inputs, **kwargs):
        embeddings = self._extract_object_embeddings(inputs["object_ids"], inputs["object_types"])
        return self._maybe_add_position_embeddings(inputs, embeddings)
