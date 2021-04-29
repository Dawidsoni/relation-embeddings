from abc import abstractmethod
import tensorflow as tf
import gin.tf
from dataclasses import dataclass

from layers.embeddings_layers import EmbeddingsConfig
from models.knowledge_completion_model import KnowledgeCompletionModel


@gin.configurable
@dataclass
class ConvModelConfig(object):
    include_reduce_dim_layer: bool = False
    normalize_embeddings: bool = False
    dropout_rate: float = 0.0


class ConvBaseModel(KnowledgeCompletionModel):

    def __init__(self, embeddings_config: EmbeddingsConfig, model_config: ConvModelConfig):
        super(ConvBaseModel, self).__init__(embeddings_config)
        self.normalize_embeddings = model_config.normalize_embeddings
        self.dropout_layer = tf.keras.layers.Dropout(model_config.dropout_rate)
        self.reduce_layer = (
            tf.keras.layers.Dense(units=1, activation=None) if model_config.include_reduce_dim_layer else None
        )

    @property
    @abstractmethod
    def conv_layers(self):
        pass

    def _rate_triples(self, head_entity_embeddings, relation_embeddings, tail_entity_embeddings, training=None):
        embeddings_to_concat = [head_entity_embeddings, relation_embeddings, tail_entity_embeddings]
        image_of_embeddings = tf.expand_dims(tf.concat(embeddings_to_concat, axis=2), axis=3)
        conv_outputs = []
        for conv_layer in self.conv_layers:
            expanded_conv_output = conv_layer(image_of_embeddings)
            flat_conv_output = tf.reshape(expanded_conv_output, (tf.shape(expanded_conv_output)[0], -1))
            conv_outputs.append(flat_conv_output)
        concat_output = tf.concat(conv_outputs, axis=1)
        dropout_output = self.dropout_layer(concat_output, training=training)
        if self.reduce_layer is None:
            return dropout_output
        return self.reduce_layer(dropout_output)

    def _transform_and_rate_embeddings(self, embeddings, training=None):
        head_entity_embeddings, relation_embeddings, tail_entity_embeddings = tf.unstack(embeddings, axis=1)
        if self.normalize_embeddings:
            head_entity_embeddings = tf.math.l2_normalize(head_entity_embeddings, axis=1)
            relation_embeddings = tf.math.l2_normalize(relation_embeddings, axis=1)
            tail_entity_embeddings = tf.math.l2_normalize(tail_entity_embeddings, axis=1)
        return self._rate_triples(
            tf.expand_dims(head_entity_embeddings, axis=2),
            tf.expand_dims(relation_embeddings, axis=2),
            tf.expand_dims(tail_entity_embeddings, axis=2),
            training,
        )

    def call(self, inputs, training=None, **kwargs):
        return self._transform_and_rate_embeddings(self.embeddings_layer(inputs), training=training)
