import tensorflow as tf
import gin.tf
import dataclasses
from dataclasses import dataclass

from layers.embeddings_layers import EmbeddingsConfig
from models.conv_base_model import ConvModelConfig
from models.transe_model import TranseModel


@gin.configurable
@dataclass
class EmbeddingsTransformConfig(object):
    constrain_embeddings_norm: bool
    constrain_transformed_embeddings_norm: bool
    initialize_transformations_with_identity: bool
    trainable_transformations_min_iteration: int


class EmbeddingsTransformInitializer(tf.keras.initializers.Initializer):

    def __init__(self, initialize_transformations_with_identity: bool):
        self.initialize_transformations_with_identity = initialize_transformations_with_identity
        self.default_initializer = tf.keras.initializers.GlorotNormal()

    def __call__(self, shape, **kwargs):
        if len(shape) < 2:
            raise ValueError(f"Expected a Tensor of rank at least 2, got {tf.rank(shape)}")
        if shape[-2] != shape[-1]:
            raise ValueError(f"Expected two last dimensions to be equal, got {shape[-2]} != {shape[-1]}")
        if self.initialize_transformations_with_identity:
            return tf.eye(shape[-1], batch_shape=shape[:-2])
        else:
            return self.default_initializer(shape)

    def get_config(self):
        return {"initialize_transformations_with_identity": self.initialize_transformations_with_identity}


@gin.configurable(blacklist=['embeddings_config'])
class STranseModel(TranseModel):

    def __init__(
        self,
        embeddings_config: EmbeddingsConfig,
        model_config: ConvModelConfig = gin.REQUIRED,
        embeddings_transform_config: EmbeddingsTransformConfig = gin.REQUIRED
    ):
        super().__init__(embeddings_config, model_config)
        self.embeddings_transform_config = embeddings_transform_config
        transformations_shape = (
            embeddings_config.relations_count,
            embeddings_config.embeddings_dimension,
            embeddings_config.embeddings_dimension,
        )
        transformations_initializer = EmbeddingsTransformInitializer(
            self.embeddings_transform_config.initialize_transformations_with_identity
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

    def get_trainable_variables_at_training_step(self, training_step):
        if self.embeddings_transform_config.trainable_transformations_min_iteration <= training_step:
            return self.trainable_variables
        non_trainable_patterns = {"head_transformation_matrices", "tail_transformation_matrices"}
        return [
            variable for variable in self.trainable_variables
            if all([pattern not in variable.name for pattern in non_trainable_patterns])
        ]

    def get_config(self):
        return {
            **dataclasses.asdict(self.embeddings_config),
            **dataclasses.asdict(self.model_config),
            **dataclasses.asdict(self.embeddings_transform_config),
        }

    def call(self, inputs, training=None, **kwargs):
        extracted_embeddings = self.embeddings_layer(inputs)
        head_entity_embeddings, relation_embeddings, tail_entity_embeddings = tf.unstack(extracted_embeddings, axis=1)
        if self.normalize_embeddings:
            head_entity_embeddings = tf.math.l2_normalize(head_entity_embeddings, axis=1)
            relation_embeddings = tf.math.l2_normalize(relation_embeddings, axis=1)
            tail_entity_embeddings = tf.math.l2_normalize(tail_entity_embeddings, axis=1)
        relation_ids = tf.unstack(inputs["object_ids"], axis=1)[1]
        head_entity_embeddings = self._constrain_embeddings(tf.expand_dims(head_entity_embeddings, axis=2))
        head_transformations = tf.gather(self.head_transformation_matrices, relation_ids)
        head_entity_embeddings = tf.linalg.matmul(head_transformations, head_entity_embeddings)
        head_entity_embeddings = self._constrain_transformed_embeddings(head_entity_embeddings)
        tail_transformations = tf.gather(self.tail_transformation_matrices, relation_ids)
        tail_entity_embeddings = self._constrain_embeddings(tf.expand_dims(tail_entity_embeddings, axis=2))
        tail_entity_embeddings = tf.linalg.matmul(tail_transformations, tail_entity_embeddings)
        tail_entity_embeddings = self._constrain_transformed_embeddings(tail_entity_embeddings)
        relation_embeddings = self._constrain_embeddings(tf.expand_dims(relation_embeddings, axis=2))
        return self._rate_triples(head_entity_embeddings, relation_embeddings, tail_entity_embeddings, training)
