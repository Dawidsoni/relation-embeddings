import numpy as np
import tensorflow as tf
import gin.tf

from models.conv_base_model import EmbeddingsConfig, ConvModelConfig
from models.transe_model import TranseModel
from layers.embeddings_layers import ObjectType


class TestTranseModel(tf.test.TestCase):

    def setUp(self):
        self.entity_embeddings = np.array([[0., 0., 0., 0.], [1., 1., 2., 1.], [2., 2., 2., 2.]], dtype=np.float32)
        self.relation_embeddings = np.array([[3., 3., 3., 3.], [4., 3., 4., 4.]], dtype=np.float32)
        edge_object_type = [ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value]
        self.model_inputs = {
            "object_ids": tf.constant([[0, 0, 1], [1, 1, 2]], dtype=tf.int32),
            "object_types": tf.constant([edge_object_type, edge_object_type], dtype=np.int32),
        }
        gin.clear_config()

    def test_outputs(self):
        embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=4)
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        transe_model = TranseModel(embeddings_config, model_config)
        transe_model.embeddings_layer.entity_embeddings.assign(self.entity_embeddings)
        transe_model.embeddings_layer.relation_embeddings.assign(self.relation_embeddings)
        outputs = transe_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[2., 2., 1., 2.], [3., 2., 4., 3.]], outputs.numpy())

    def test_reduce_dim_layer(self):
        embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=4)
        model_config = ConvModelConfig(include_reduce_dim_layer=True)
        transe_model = TranseModel(embeddings_config, model_config)
        transe_model.embeddings_layer.entity_embeddings.assign(self.entity_embeddings)
        transe_model.embeddings_layer.relation_embeddings.assign(self.relation_embeddings)
        transe_model(self.model_inputs, training=True)
        transe_model.reduce_layer.set_weights([np.array([[1.], [2.], [1.], [0.]]), np.array([-10.])])
        output = transe_model(self.model_inputs, training=True)
        self.assertEqual((2, 1), output.shape)
        self.assertAllClose([[-3.], [1.]], output.numpy())

    def test_pretrained_embeddings(self):
        embeddings_config = EmbeddingsConfig(
            entities_count=3, relations_count=2, pretrained_entity_embeddings=self.entity_embeddings,
            embeddings_dimension=4, pretrained_relation_embeddings=self.relation_embeddings
        )
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        transe_model = TranseModel(embeddings_config, model_config)
        outputs = transe_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[2., 2., 1., 2.], [3., 2., 4., 3.]], outputs.numpy())

    def test_trainable_embeddings(self):
        embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=4)
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        transe_model = TranseModel(embeddings_config, model_config)
        transe_model.embeddings_layer.entity_embeddings.assign(self.entity_embeddings)
        transe_model.embeddings_layer.relation_embeddings.assign(self.relation_embeddings)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        with tf.GradientTape() as gradient_tape:
            model_outputs = transe_model(self.model_inputs, training=True)
            loss = tf.keras.losses.mean_squared_error(tf.zeros_like(model_outputs), model_outputs)
        gradients = gradient_tape.gradient(loss, transe_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transe_model.trainable_variables))
        self.assertGreater(np.sum(self.entity_embeddings != transe_model.embeddings_layer.entity_embeddings.numpy()), 0)

    def test_non_trainable_embeddings(self):
        embeddings_config = EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4, trainable_embeddings=False
        )
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        transe_model = TranseModel(embeddings_config, model_config)
        transe_model.embeddings_layer.entity_embeddings.assign(self.entity_embeddings)
        transe_model.embeddings_layer.relation_embeddings.assign(self.relation_embeddings)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        with tf.GradientTape() as gradient_tape:
            model_outputs = transe_model(self.model_inputs, training=True)
            loss = tf.keras.losses.mean_squared_error(tf.zeros_like(model_outputs), model_outputs)
        gradients = gradient_tape.gradient(loss, transe_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transe_model.trainable_variables))
        self.assertEqual(np.sum(self.entity_embeddings != transe_model.embeddings_layer.entity_embeddings.numpy()), 0)

    def test_non_trainable_convolutions(self):
        embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=4)
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        transe_model = TranseModel(embeddings_config, model_config)
        transe_model.embeddings_layer.entity_embeddings.assign(self.entity_embeddings)
        transe_model.embeddings_layer.relation_embeddings.assign(self.relation_embeddings)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        with tf.GradientTape() as gradient_tape:
            model_outputs = transe_model(self.model_inputs, training=True)
            loss = tf.keras.losses.mean_squared_error(tf.zeros_like(model_outputs), model_outputs)
        gradients = gradient_tape.gradient(loss, transe_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transe_model.trainable_variables))
        expected_kernel = np.array([[[[1.0]], [[1.0]], [[-1.0]]]])
        self.assertAllEqual(expected_kernel, transe_model.conv_layers[0].kernel.numpy())
        expected_bias = np.array([0.0])
        self.assertAllEqual(expected_bias, transe_model.conv_layers[0].bias.numpy())

    def test_normalize_embeddings(self):
        embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=2)
        model_config = ConvModelConfig(include_reduce_dim_layer=False, normalize_embeddings=True)
        transe_model = TranseModel(embeddings_config, model_config)
        transe_model.embeddings_layer.entity_embeddings.assign([[0., 0.], [1., 0.], [0.8, 0.6]])
        transe_model.embeddings_layer.relation_embeddings.assign([[1.6, 1.2], [2.4, 3.2]])
        outputs = transe_model(self.model_inputs, training=True)
        self.assertEqual((2, 2), outputs.shape)
        self.assertAllClose([[-0.2, 0.6], [0.8, 0.2]], outputs.numpy())

    def test_dropout_layer(self):
        tf.random.set_seed(1)
        embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=4)
        model_config = ConvModelConfig(include_reduce_dim_layer=False, dropout_rate=0.999)
        transe_model = TranseModel(embeddings_config, model_config)
        transe_model.embeddings_layer.entity_embeddings.assign(self.entity_embeddings)
        transe_model.embeddings_layer.relation_embeddings.assign(self.relation_embeddings)
        outputs = transe_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose(tf.zeros_like(outputs), outputs)

    def test_gin_config(self):
        gin_config = """
            ConvModelConfig.include_reduce_dim_layer = False
            ConvModelConfig.normalize_embeddings = False
            ConvModelConfig.dropout_rate = 0.0
            TranseModel.model_config = @ConvModelConfig()
        """
        gin.parse_config(gin_config)
        embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=4)
        transe_model = TranseModel(embeddings_config)
        transe_model.embeddings_layer.entity_embeddings.assign(self.entity_embeddings)
        transe_model.embeddings_layer.relation_embeddings.assign(self.relation_embeddings)
        outputs = transe_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[2., 2., 1., 2.], [3., 2., 4., 3.]], outputs.numpy())


if __name__ == '__main__':
    tf.test.main()
