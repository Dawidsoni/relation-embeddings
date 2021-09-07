import dataclasses
from dataclasses import dataclass
import gin.tf
import tensorflow as tf

from layers.embeddings_layers import EmbeddingsConfig
from layers.transformer_layers import StackedTransformerEncodersLayer
from models.knowledge_completion_model import EmbeddingsBasedModel
from optimization import parameters_factory


@gin.configurable
@dataclass
class TransformerSoftmaxModelConfig(object):
    use_pre_normalization: bool
    pre_dropout_rate: float
    use_relations_outputs: bool = False


@gin.configurable(blacklist=['embeddings_config'])
class TransformerSoftmaxModel(EmbeddingsBasedModel):

    def __init__(self, embeddings_config: EmbeddingsConfig, model_config: TransformerSoftmaxModelConfig = gin.REQUIRED):
        super().__init__(embeddings_config)
        self.model_config = model_config
        self.pre_normalization_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pre_dropout_layer = tf.keras.layers.Dropout(rate=self.model_config.pre_dropout_rate)
        self.transformer_layer = StackedTransformerEncodersLayer()
        self.post_hidden_layer = tf.keras.layers.Dense(
            units=self.embeddings_layer.config.embeddings_dimension,
            kernel_initializer=parameters_factory.get_parameters_initializer(),
        )
        self.post_normalization_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.projection_bias = tf.Variable(
            initial_value=tf.zeros_initializer()(shape=(tf.shape(self.get_similarity_matrix())[0], )),
            trainable=True,
        )

    def get_similarity_matrix(self):
        if self.model_config.use_relations_outputs:
            return tf.concat(
                [self.embeddings_layer.entity_embeddings, self.embeddings_layer.relation_embeddings],
                axis=0,
            )
        return self.embeddings_layer.entity_embeddings

    def projection_layer(self, inputs, training):
        outputs = tf.linalg.matmul(inputs, self.get_similarity_matrix(), transpose_b=True)
        outputs += self.projection_bias
        return outputs

    def call(self, inputs, training=None, apply_projection_layer=True, **kwargs):
        outputs = self.embeddings_layer(inputs, training=training)
        if self.model_config.use_pre_normalization:
            outputs = self.pre_normalization_layer(outputs, training=training)
        outputs = self.pre_dropout_layer(outputs, training=training)
        outputs = self.transformer_layer(outputs, training=training)
        outputs = tf.gather(outputs, indices=inputs["mask_index"], axis=1, batch_dims=1)
        outputs = self.post_hidden_layer(outputs, training=training)
        if not apply_projection_layer:
            return outputs
        return self.projection_layer(outputs, training=training)

    def get_config(self):
        return {
            **dataclasses.asdict(self.embeddings_config),
            **dataclasses.asdict(self.model_config),
        }
