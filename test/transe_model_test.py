import numpy as np
import tensorflow as tf
import gin.tf

from conv_base_model import DataConfig, ModelConfig
from transe_model import TranseModel


class TestTranseModel(tf.test.TestCase):

    def setUp(self):
        self.entity_embeddings = np.array([[0., 0., 0., 0.], [1., 1., 2., 1.], [2., 2., 2., 2.]], dtype=np.float32)
        self.relation_embeddings = np.array([[3., 3., 3., 3.], [4., 3., 4., 4.]], dtype=np.float32)
        self.model_inputs = np.array([[0, 0, 1], [1, 1, 2]])

    def test_outputs(self):
        data_config = DataConfig(entities_count=3, relations_count=2)
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False)
        transe_model = TranseModel(data_config, model_config)
        transe_model.entity_embeddings.assign(self.entity_embeddings)
        transe_model.relation_embeddings.assign(self.relation_embeddings)
        outputs = transe_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[2., 2., 1., 2.], [3., 2., 4., 3.]], outputs.numpy())

    def test_reduce_dim_layer(self):
        data_config = DataConfig(entities_count=3, relations_count=2)
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=True)
        transe_model = TranseModel(data_config, model_config)
        transe_model.entity_embeddings.assign(self.entity_embeddings)
        transe_model.relation_embeddings.assign(self.relation_embeddings)
        transe_model(self.model_inputs, training=True)
        transe_model.reduce_layer.set_weights([np.array([[1.], [2.], [1.], [0.]]), np.array([-10.])])
        output = transe_model(self.model_inputs, training=True)
        self.assertEqual((2, 1), output.shape)
        self.assertAllClose([[-3.], [1.]], output.numpy())

    def test_pretrained_embeddings(self):
        data_config = DataConfig(
            entities_count=3, relations_count=2, pretrained_entity_embeddings=self.entity_embeddings,
            pretrained_relations_embeddings=self.relation_embeddings
        )
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False)
        transe_model = TranseModel(data_config, model_config)
        outputs = transe_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[2., 2., 1., 2.], [3., 2., 4., 3.]], outputs.numpy())

    def test_trainable_embeddings(self):
        data_config = DataConfig(entities_count=3, relations_count=2)
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False)
        transe_model = TranseModel(data_config, model_config)
        transe_model.entity_embeddings.assign(self.entity_embeddings)
        transe_model.relation_embeddings.assign(self.relation_embeddings)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        with tf.GradientTape() as gradient_tape:
            model_outputs = transe_model(self.model_inputs, training=True)
            loss = tf.keras.losses.mean_squared_error(tf.zeros_like(model_outputs), model_outputs)
        gradients = gradient_tape.gradient(loss, transe_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transe_model.trainable_variables))
        self.assertGreater(np.sum(self.entity_embeddings != transe_model.entity_embeddings.numpy()), 0)

    def test_non_trainable_embeddings(self):
        data_config = DataConfig(entities_count=3, relations_count=2)
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False, trainable_embeddings=False)
        transe_model = TranseModel(data_config, model_config)
        transe_model.entity_embeddings.assign(self.entity_embeddings)
        transe_model.relation_embeddings.assign(self.relation_embeddings)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        with tf.GradientTape() as gradient_tape:
            model_outputs = transe_model(self.model_inputs, training=True)
            loss = tf.keras.losses.mean_squared_error(tf.zeros_like(model_outputs), model_outputs)
        gradients = gradient_tape.gradient(loss, transe_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transe_model.trainable_variables))
        self.assertEqual(np.sum(self.entity_embeddings != transe_model.entity_embeddings.numpy()), 0)

    def test_non_trainable_convolutions(self):
        data_config = DataConfig(entities_count=3, relations_count=2)
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False, trainable_embeddings=False)
        transe_model = TranseModel(data_config, model_config)
        transe_model.entity_embeddings.assign(self.entity_embeddings)
        transe_model.relation_embeddings.assign(self.relation_embeddings)
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
        data_config = DataConfig(entities_count=3, relations_count=2)
        model_config = ModelConfig(embeddings_dimension=2, include_reduce_dim_layer=False, normalize_embeddings=True)
        transe_model = TranseModel(data_config, model_config)
        transe_model.entity_embeddings.assign([[0., 0.], [1., 0.], [0.8, 0.6]])
        transe_model.relation_embeddings.assign([[1.6, 1.2], [2.4, 3.2]])
        outputs = transe_model([[0, 0, 1], [1, 1, 2]], training=True)
        self.assertEqual((2, 2), outputs.shape)
        self.assertAllClose([[-0.2, 0.6], [0.8, 0.2]], outputs.numpy())

    def test_dropout_layer(self):
        tf.random.set_seed(1)
        data_config = DataConfig(entities_count=3, relations_count=2)
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False, dropout_rate=0.999)
        transe_model = TranseModel(data_config, model_config)
        transe_model.entity_embeddings.assign(self.entity_embeddings)
        transe_model.relation_embeddings.assign(self.relation_embeddings)
        outputs = transe_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose(tf.zeros_like(outputs), outputs)

    def test_gin_config(self):
        gin_config = """
            ModelConfig.embeddings_dimension = 4
            ModelConfig.trainable_embeddings = True
            ModelConfig.include_reduce_dim_layer = False
            ModelConfig.normalize_embeddings = False
            ModelConfig.dropout_rate = 0.0
            TranseModel.model_config = @ModelConfig()
        """
        gin.parse_config(gin_config)
        data_config = DataConfig(entities_count=3, relations_count=2)
        transe_model = TranseModel(data_config)
        transe_model.entity_embeddings.assign(self.entity_embeddings)
        transe_model.relation_embeddings.assign(self.relation_embeddings)
        outputs = transe_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[2., 2., 1., 2.], [3., 2., 4., 3.]], outputs.numpy())


if __name__ == '__main__':
    tf.test.main()
