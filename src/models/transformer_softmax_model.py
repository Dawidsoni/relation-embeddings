import dataclasses
from dataclasses import dataclass
import gin.tf
import tensorflow as tf

from layers.embeddings_layers import EmbeddingsConfig
from layers.transformer_layers import StackedTransformerEncodersLayer
from models.knowledge_completion_model import KnowledgeCompletionModel
from optimization import parameters_factory


@gin.configurable
@dataclass
class TransformerSoftmaxModelConfig(object):
    use_pre_normalization: bool
    pre_dropout_rate: float
    use_sigmoid_as_output_layer: float


@gin.configurable(blacklist=['embeddings_config'])
class TransformerSoftmaxModel(KnowledgeCompletionModel):

    def __init__(self, embeddings_config: EmbeddingsConfig, model_config: TransformerSoftmaxModelConfig = gin.REQUIRED):
        super().__init__(embeddings_config)
        self.model_config = model_config
        self.pre_normalization_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pre_dropout_layer = tf.keras.layers.Dropout(rate=self.model_config.pre_dropout_rate)
        self.transformer_layer = StackedTransformerEncodersLayer()
        self.post_hidden_layer = tf.keras.layers.Dense(
            units=self.embeddings_layer.config.embeddings_dimension,
            activation=parameters_factory.get_activation(),
            kernel_initializer=parameters_factory.get_parameters_initializer(),
        )
        self.post_normalization_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.projection_bias = tf.Variable(
            initial_value=tf.zeros_initializer()(shape=(self.embeddings_layer.config.entities_count, )),
            trainable=True,
        )

    def call(self, inputs, training=None, **kwargs):
        outputs = self.embeddings_layer(inputs, training=training)
        if self.model_config.use_pre_normalization:
            outputs = self.pre_normalization_layer(outputs, training=training)
        outputs = self.pre_dropout_layer(outputs, training=training)
        outputs = self.transformer_layer(outputs, training=training)
        outputs = tf.gather(outputs, indices=inputs["mask_index"], axis=1, batch_dims=1)
        outputs = self.post_hidden_layer(outputs, training=training)
        outputs = self.post_normalization_layer(outputs, training=training)
        outputs = tf.linalg.matmul(outputs, self.embeddings_layer.entity_embeddings, transpose_b=True)
        outputs += self.projection_bias
        return outputs

    def get_config(self):
        return {
            **dataclasses.asdict(self.embeddings_config),
            **dataclasses.asdict(self.model_config),
        }
