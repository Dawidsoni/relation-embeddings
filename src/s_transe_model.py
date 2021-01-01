import tensorflow as tf
import gin.tf
from dataclasses import dataclass

from conv_base_model import DataConfig, ModelConfig
from transe_model import TranseModel


@gin.configurable
@dataclass
class EmbeddingsTransformConfig:
    constrain_embeddings_norm: bool
    constrain_transformed_embeddings_norm: bool
    initialize_with_identity: bool


class EmbeddingsTransformInitializer(tf.keras.initializers.Initializer):

    def __init__(self, initialize_with_identity: bool):
        self.initialize_with_identity = initialize_with_identity
        self.default_initializer = tf.keras.initializers.GlorotNormal()

    def __call__(self, shape, **kwargs):
        if len(shape) < 2:
            raise ValueError(f"Expected a Tensor of rank at least 2, got {tf.rank(shape)}")
        if shape[-2] != shape[-1]:
            raise ValueError(f"Expected two last dimensions to be equal, got {shape[-2]} != {shape[-1]}")
        if self.initialize_with_identity:
            return tf.eye(shape[-1], batch_shape=shape[:-2])
        else:
            return self.default_initializer(shape)

    def get_config(self):
        return {"initialize_with_identity": self.initialize_with_identity}


@gin.configurable(blacklist=['data_config'])
class STranseModel(TranseModel):

    def __init__(
        self, data_config: DataConfig, model_config: ModelConfig = gin.REQUIRED,
        embeddings_transform_config: EmbeddingsTransformConfig = gin.REQUIRED
    ):
        super().__init__(data_config, model_config)
        self.embeddings_transform_config = embeddings_transform_config
        transformations_shape = (
            data_config.relations_count, model_config.embeddings_dimension, model_config.embeddings_dimension
        )
        transformations_initializer = EmbeddingsTransformInitializer(
            self.embeddings_transform_config.initialize_with_identity
        )
        self.head_transformation_matrices = tf.Variable(
            initial_value=transformations_initializer(transformations_shape),
            name="head_transformation_matrices",
            trainable=True,
        )
        self.tail_transformation_matrices = tf.Variable(
            initial_value=transformations_initializer(transformations_shape),
            name="tail_transformation_matrices",
            trainable=True
        )

    def _constrain_embeddings(self, embeddings: tf.Tensor):
        if not self.embeddings_transform_config.constrain_embeddings_norm:
            return embeddings
        constrained_norms = tf.maximum(tf.norm(embeddings, axis=1), 1.0)
        return embeddings / tf.expand_dims(constrained_norms, axis=2)

    def _constrain_transformed_embeddings(self, embeddings: tf.Tensor):
        if not self.embeddings_transform_config.constrain_transformed_embeddings_norm:
            return embeddings
        constrained_norms = tf.maximum(tf.norm(embeddings, axis=1), 1.0)
        return embeddings / tf.expand_dims(constrained_norms, axis=2)

    def extract_and_transform_embeddings(self, head_entity_ids, relation_ids, tail_entity_ids):
        head_entity_embeddings, relation_embeddings, tail_entity_embeddings = super().extract_and_transform_embeddings(
            head_entity_ids, relation_ids, tail_entity_ids
        )
        head_entity_embeddings = self._constrain_embeddings(head_entity_embeddings)
        relation_embeddings = self._constrain_embeddings(relation_embeddings)
        tail_entity_embeddings = self._constrain_embeddings(tail_entity_embeddings)
        head_transformations = tf.gather(self.head_transformation_matrices, relation_ids)
        tail_transformations = tf.gather(self.tail_transformation_matrices, relation_ids)
        head_entity_embeddings = tf.linalg.matmul(head_transformations, head_entity_embeddings)
        tail_entity_embeddings = tf.linalg.matmul(tail_transformations, tail_entity_embeddings)
        head_entity_embeddings = self._constrain_transformed_embeddings(head_entity_embeddings)
        tail_entity_embeddings = self._constrain_transformed_embeddings(tail_entity_embeddings)
        return head_entity_embeddings, relation_embeddings, tail_entity_embeddings
