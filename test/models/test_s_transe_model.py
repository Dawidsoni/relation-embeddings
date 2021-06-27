import numpy as np
import tensorflow as tf
import gin.tf

from models.conv_base_model import EmbeddingsConfig, ConvModelConfig
from models.s_transe_model import STranseModel, EmbeddingsTransformConfig
from layers.embeddings_layers import ObjectType


class STestTranseModel(tf.test.TestCase):

    def setUp(self):
        self.entity_embeddings = np.array([[0., 0., 0., 0.], [0.1, 0.2, 0.3, 0.4], [3., 4., 0., 0.]], dtype=np.float32)
        self.relation_embeddings = np.array([[3., 3., 3., 3.], [0.6, -0.8, 0.0, 0.0]], dtype=np.float32)
        self.embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=4)
        self.model_config = ConvModelConfig(include_reduce_dim_layer=False)
        edge_object_type = [ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value]
        self.model_inputs = {
            "object_ids": tf.constant([[0, 0, 1], [0, 1, 2]], dtype=tf.int32),
            "object_types": tf.constant([edge_object_type, edge_object_type], dtype=tf.int32),
        }
        gin.clear_config()

    def test_outputs(self):
        embeddings_transform_config = EmbeddingsTransformConfig(
            constrain_embeddings_norm=False, constrain_transformed_embeddings_norm=False,
            initialize_transformations_with_identity=False, trainable_transformations_min_iteration=0
        )
        model = STranseModel(self.embeddings_config, self.model_config, embeddings_transform_config)
        model.embeddings_layer.entity_embeddings.assign(self.entity_embeddings)
        model.embeddings_layer.relation_embeddings.assign(self.relation_embeddings)
        model.head_transformation_matrices.assign([tf.eye(4), -2 * tf.eye(4)])
        model.tail_transformation_matrices.assign([-2 * tf.eye(4), 2 * tf.eye(4)])
        outputs = model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[3.2, 3.4, 3.6, 3.8], [-5.4, -8.8, 0., 0.]], outputs.numpy())

    def test_constrain_embeddings_norm(self):
        embeddings_transform_config = EmbeddingsTransformConfig(
            constrain_embeddings_norm=True, constrain_transformed_embeddings_norm=False,
            initialize_transformations_with_identity=False, trainable_transformations_min_iteration=0
        )
        model = STranseModel(self.embeddings_config, self.model_config, embeddings_transform_config)
        model.embeddings_layer.entity_embeddings.assign(self.entity_embeddings)
        model.embeddings_layer.relation_embeddings.assign(self.relation_embeddings)
        model.head_transformation_matrices.assign([tf.eye(4), -2 * tf.eye(4)])
        model.tail_transformation_matrices.assign([tf.eye(4), -2 * tf.eye(4)])
        outputs = model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[0.4, 0.3, 0.2, 0.1], [1.8, 0.8, 0., 0.]], outputs.numpy())

    def test_constrain_transformed_embeddings_norm(self):
        embeddings_transform_config = EmbeddingsTransformConfig(
            constrain_embeddings_norm=False, constrain_transformed_embeddings_norm=True,
            initialize_transformations_with_identity=False, trainable_transformations_min_iteration=0
        )
        model = STranseModel(self.embeddings_config, self.model_config, embeddings_transform_config)
        model.embeddings_layer.entity_embeddings.assign(self.entity_embeddings)
        model.embeddings_layer.relation_embeddings.assign(self.relation_embeddings)
        model.head_transformation_matrices.assign([tf.eye(4), -2 * tf.eye(4)])
        model.tail_transformation_matrices.assign([tf.eye(4), -2 * tf.eye(4)])
        outputs = model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[2.9, 2.8, 2.7, 2.6], [1.2, 0.0, 0.0, 0.0]], outputs.numpy())

    def test_initialize_transformations_with_identity(self):
        embeddings_transform_config = EmbeddingsTransformConfig(
            constrain_embeddings_norm=False, constrain_transformed_embeddings_norm=False,
            initialize_transformations_with_identity=True, trainable_transformations_min_iteration=0
        )
        model = STranseModel(self.embeddings_config, self.model_config, embeddings_transform_config)
        model.embeddings_layer.entity_embeddings.assign(self.entity_embeddings)
        model.embeddings_layer.relation_embeddings.assign(self.relation_embeddings)
        outputs = model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[2.9, 2.8, 2.7, 2.6], [-2.4, -4.8, 0.0, 0.0]], outputs.numpy())

    def test_training_step_less_than_transformations_min_iteration(self):
        embeddings_transform_config = EmbeddingsTransformConfig(
            constrain_embeddings_norm=False, constrain_transformed_embeddings_norm=False,
            initialize_transformations_with_identity=True, trainable_transformations_min_iteration=2
        )
        model = STranseModel(self.embeddings_config, self.model_config, embeddings_transform_config)
        trainable_variables = model.get_trainable_variables_at_training_step(training_step=1)
        self.assertLen(trainable_variables, expected_len=2)

    def test_training_step_equal_transformations_min_iteration(self):
        embeddings_transform_config = EmbeddingsTransformConfig(
            constrain_embeddings_norm=False, constrain_transformed_embeddings_norm=False,
            initialize_transformations_with_identity=True, trainable_transformations_min_iteration=2
        )
        model = STranseModel(self.embeddings_config, self.model_config, embeddings_transform_config)
        trainable_variables = model.get_trainable_variables_at_training_step(training_step=2)
        self.assertLen(trainable_variables, expected_len=4)


if __name__ == '__main__':
    tf.test.main()
