from abc import abstractmethod
import tensorflow as tf


class ConvBaseModel(tf.keras.Model):

    def __init__(
        self, entities_count, relations_count, embedding_dimension, include_reduce_dim_layer=False,
        pretrained_entity_embeddings=None, pretrained_relation_embeddings=None, trainable_embeddings=True,
        normalize_embeddings=False, dropout_rate=0.0
    ):
        super(ConvBaseModel, self).__init__()
        self.entity_embeddings = tf.Variable(
            self._get_initial_embedding_values([entities_count, embedding_dimension], pretrained_entity_embeddings),
            name='entity_embeddings',
            trainable=trainable_embeddings
        )
        self.relation_embeddings = tf.Variable(
            self._get_initial_embedding_values([relations_count, embedding_dimension], pretrained_relation_embeddings),
            name='relation_embeddings',
            trainable=trainable_embeddings
        )
        self.normalize_embeddings = normalize_embeddings
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.reduce_layer = tf.keras.layers.Dense(units=1, activation=None) if include_reduce_dim_layer else None

    @staticmethod
    def _get_initial_embedding_values(shape, pretrained_embeddings=None):
        if pretrained_embeddings is None:
            return tf.random.truncated_normal(shape=shape)
        return pretrained_embeddings

    @property
    @abstractmethod
    def conv_layers(self):
        pass

    def call(self, inputs):
        head_entity_ids, relation_ids, tail_entity_ids = tf.unstack(inputs, axis=1)
        head_entity_embeddings = tf.expand_dims(tf.gather(self.entity_embeddings, head_entity_ids), axis=2)
        relation_embeddings = tf.expand_dims(tf.gather(self.relation_embeddings, relation_ids), axis=2)
        tail_entity_embeddings = tf.expand_dims(tf.gather(self.entity_embeddings, tail_entity_ids), axis=2)
        if self.normalize_embeddings:
            head_entity_embeddings = tf.math.l2_normalize(head_entity_embeddings, axis=1)
            relation_embeddings = tf.math.l2_normalize(relation_embeddings, axis=1)
            tail_entity_embeddings = tf.math.l2_normalize(tail_entity_embeddings, axis=1)
        concat_embeddings = tf.concat([head_entity_embeddings, relation_embeddings, tail_entity_embeddings], axis=2)
        image_of_embeddings = tf.expand_dims(concat_embeddings, axis=3)
        conv_outputs = []
        for conv_layer in self.conv_layers:
            expanded_conv_output = conv_layer(image_of_embeddings)
            flat_conv_output = tf.reshape(expanded_conv_output, (tf.shape(expanded_conv_output)[0], -1))
            conv_outputs.append(flat_conv_output)
        concat_output = tf.concat(conv_outputs, axis=1)
        dropout_output = self.dropout_layer(concat_output)
        if self.reduce_layer is None:
            return dropout_output
        return self.reduce_layer(dropout_output)
