from abc import abstractmethod
import tensorflow as tf
import gin.tf
import numpy as np
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class DataConfig:
    entities_count: int
    relations_count: int
    pretrained_entity_embeddings: Optional[np.ndarray] = None
    pretrained_relations_embeddings: Optional[np.ndarray] = None


@gin.configurable
@dataclass
class ModelConfig:
    embeddings_dimension: int
    trainable_embeddings: bool = True
    include_reduce_dim_layer: bool = False
    normalize_embeddings: bool = False
    dropout_rate: float = 0.0


class ConvBaseModel(tf.keras.Model):

    def __init__(self, data_config: DataConfig, model_config: ModelConfig):
        super(ConvBaseModel, self).__init__()
        entities_shape = [data_config.entities_count, model_config.embeddings_dimension]
        self.entity_embeddings = tf.Variable(
            self._get_initial_embedding_values(entities_shape, data_config.pretrained_entity_embeddings),
            name='entity_embeddings',
            trainable=model_config.trainable_embeddings
        )
        relations_shape = [data_config.relations_count, model_config.embeddings_dimension]
        self.relation_embeddings = tf.Variable(
            self._get_initial_embedding_values(relations_shape, data_config.pretrained_relations_embeddings),
            name='relation_embeddings',
            trainable=model_config.trainable_embeddings
        )
        self.normalize_embeddings = model_config.normalize_embeddings
        self.dropout_layer = tf.keras.layers.Dropout(model_config.dropout_rate)
        self.reduce_layer = (
            tf.keras.layers.Dense(units=1, activation=None) if model_config.include_reduce_dim_layer else None
        )

    @staticmethod
    def _get_initial_embedding_values(shape, pretrained_embeddings=None):
        if pretrained_embeddings is None:
            return tf.random.truncated_normal(shape=shape)
        return pretrained_embeddings

    def extract_and_transform_embeddings(self, head_entity_ids, relation_ids, tail_entity_ids):
        head_entity_embeddings = tf.expand_dims(tf.gather(self.entity_embeddings, head_entity_ids), axis=2)
        relation_embeddings = tf.expand_dims(tf.gather(self.relation_embeddings, relation_ids), axis=2)
        tail_entity_embeddings = tf.expand_dims(tf.gather(self.entity_embeddings, tail_entity_ids), axis=2)
        if self.normalize_embeddings:
            head_entity_embeddings = tf.math.l2_normalize(head_entity_embeddings, axis=1)
            relation_embeddings = tf.math.l2_normalize(relation_embeddings, axis=1)
            tail_entity_embeddings = tf.math.l2_normalize(tail_entity_embeddings, axis=1)
        return head_entity_embeddings, relation_embeddings, tail_entity_embeddings

    @property
    @abstractmethod
    def conv_layers(self):
        pass

    def _rate_triples(self, image_of_embeddings, training=None):
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

    def call(self, inputs, training=None, **kwargs):
        head_entity_ids, relation_ids, tail_entity_ids = tf.unstack(inputs, axis=1)
        head_entity_embeddings, relation_embeddings, tail_entity_embeddings = self.extract_and_transform_embeddings(
            head_entity_ids, relation_ids, tail_entity_ids
        )
        concat_embeddings = tf.concat([head_entity_embeddings, relation_embeddings, tail_entity_embeddings], axis=2)
        image_of_embeddings = tf.expand_dims(concat_embeddings, axis=3)
        return self._rate_triples(image_of_embeddings, training)

    def get_trainable_variables_at_training_step(self, training_step):
        return self.trainable_variables

    def save_with_embeddings(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_weights(filepath=os.path.join(path, "saved_weights.tf"), save_format="tf")
        np.save(file=os.path.join(path, "entity_embeddings"), arr=self.entity_embeddings.numpy())
        np.save(file=os.path.join(path, "relation_embeddings"), arr=self.relation_embeddings.numpy())
