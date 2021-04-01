import numpy as np
import tensorflow as tf
import gin.tf

from models.conv_base_model import EmbeddingsConfig, ConvModelConfig
from models.convkb_model import ConvKBModel, ConvolutionsConfig


class TestConvKBModel(tf.test.TestCase):

    def _set_default_weights(self, model):
        model.embeddings_layer.entity_embeddings.assign(self.default_entity_embeddings)
        model.embeddings_layer.relation_embeddings.assign(self.default_relation_embeddings)
        model(self.model_inputs)
        model.conv_layers[0].kernel.assign(self.default_kernel)
        model.conv_layers[0].bias.assign(self.default_bias)
        gin.clear_config()

    def setUp(self):
        self.default_embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=4)
        self.default_convolutions_config = ConvolutionsConfig(
            filter_heights=[1], filters_count_per_height=1, activation="relu", use_constant_initialization=False
        )
        self.default_entity_embeddings = np.array(
            [[0., 0., 0., 0.], [1., 1., 2., 1.], [2., 2., 2., 2.]], dtype=np.float32
        )
        self.default_relation_embeddings = np.array([[3., 3., 3., 3.], [4., 3., 4., 4.]], dtype=np.float32)
        self.default_kernel = np.array([[[[1.0]], [[1.0]], [[-1.0]]]])
        self.default_bias = np.array([0.0])
        self.model_inputs = np.array([[0, 0, 1], [1, 1, 2]])

    def test_outputs(self):
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        convkb_model = ConvKBModel(self.default_embeddings_config, model_config, self.default_convolutions_config)
        self._set_default_weights(convkb_model)
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[2., 2., 1., 2.], [3., 2., 4., 3.]], outputs.numpy())

    def test_activation(self):
        embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=4)
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        convolutions_config = ConvolutionsConfig(
            filter_heights=[1], filters_count_per_height=1, activation="relu", use_constant_initialization=False
        )
        convkb_model = ConvKBModel(embeddings_config, model_config, convolutions_config)
        self._set_default_weights(convkb_model)
        convkb_model.embeddings_layer.relation_embeddings.assign(-self.default_relation_embeddings)
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose(tf.zeros_like(outputs), outputs.numpy())

    def test_filter_heights(self):
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        convolutions_config = ConvolutionsConfig(
            filter_heights=[1, 2], filters_count_per_height=1, activation="relu", use_constant_initialization=False
        )
        convkb_model = ConvKBModel(self.default_embeddings_config, model_config, convolutions_config)
        convkb_model.embeddings_layer.entity_embeddings.assign(self.default_entity_embeddings)
        convkb_model.embeddings_layer.relation_embeddings.assign(self.default_relation_embeddings)
        convkb_model(self.model_inputs)
        convkb_model.conv_layers[0].kernel.assign(self.default_kernel)
        convkb_model.conv_layers[0].bias.assign(self.default_bias)
        convkb_model.conv_layers[1].kernel.assign([[[[1.0]], [[1.0]], [[-1.0]]], [[[-1.0]], [[0.0]], [[2.0]]]])
        convkb_model.conv_layers[1].bias.assign([1.0])
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 7), outputs.shape)
        self.assertAllClose([[2., 2., 1., 2., 5., 7., 4.], [3., 2., 4., 3., 7., 5., 8.]], outputs.numpy())

    def test_filters_count_per_height(self):
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        convolutions_config = ConvolutionsConfig(
            filter_heights=[1], filters_count_per_height=2, activation="relu", use_constant_initialization=False
        )
        convkb_model = ConvKBModel(self.default_embeddings_config, model_config, convolutions_config)
        convkb_model.embeddings_layer.entity_embeddings.assign(self.default_entity_embeddings)
        convkb_model.embeddings_layer.relation_embeddings.assign(self.default_relation_embeddings)
        convkb_model(self.model_inputs)
        convkb_model.conv_layers[0].kernel.assign([[[[1.0, 2.0]], [[1.0, -1.0]], [[-1.0, 0.0]]]])
        convkb_model.conv_layers[0].bias.assign([0.0, 1.0])
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 8), outputs.shape)
        self.assertAllClose([[2., 0., 2., 0., 1., 0., 2., 0.], [3., 0., 2., 0., 4., 1., 3., 0.]], outputs.numpy())

    def test_non_constant_initialization(self):
        tf.random.set_seed(1)
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        convolutions_config = ConvolutionsConfig(
            filter_heights=[1], filters_count_per_height=1, activation="relu", use_constant_initialization=False
        )
        convkb_model = ConvKBModel(self.default_embeddings_config, model_config, convolutions_config)
        convkb_model(self.model_inputs)
        unequal_elements = convkb_model.conv_layers[0].kernel.numpy() != np.array([[[[0.1]], [[0.1]], [[-0.1]]]])
        self.assertGreater(np.sum(unequal_elements), 0)

    def test_constant_initialization(self):
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        convolutions_config = ConvolutionsConfig(
            filter_heights=[1], filters_count_per_height=1, activation="relu", use_constant_initialization=True
        )
        convkb_model = ConvKBModel(self.default_embeddings_config, model_config, convolutions_config)
        convkb_model.embeddings_layer.entity_embeddings.assign(self.default_entity_embeddings)
        convkb_model.embeddings_layer.relation_embeddings.assign(self.default_relation_embeddings)
        convkb_model(self.model_inputs)
        convkb_model.conv_layers[0].bias.assign(self.default_bias)
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[0.2, 0.2, 0.1, 0.2], [0.3, 0.2, 0.4, 0.3]], outputs.numpy())

    def test_reduce_dim_layer(self):
        model_config = ConvModelConfig(include_reduce_dim_layer=True)
        convkb_model = ConvKBModel(self.default_embeddings_config, model_config, self.default_convolutions_config)
        self._set_default_weights(convkb_model)
        convkb_model.reduce_layer.set_weights([np.array([[1.], [2.], [1.], [0.]]), np.array([-10.])])
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 1), outputs.shape)
        self.assertAllClose([[-3.], [1.]], outputs.numpy())

    def test_pretrained_embeddings(self):
        embeddings_config = EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4,
            pretrained_entity_embeddings=self.default_entity_embeddings,
            pretrained_relations_embeddings=self.default_relation_embeddings
        )
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        convkb_model = ConvKBModel(embeddings_config, model_config, self.default_convolutions_config)
        self._set_default_weights(convkb_model)
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[2., 2., 1., 2.], [3., 2., 4., 3.]], outputs.numpy())

    def test_trainable_embeddings(self):
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        convkb_model = ConvKBModel(self.default_embeddings_config, model_config, self.default_convolutions_config)
        self._set_default_weights(convkb_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        with tf.GradientTape() as gradient_tape:
            model_outputs = convkb_model(self.model_inputs, training=True)
            loss = tf.keras.losses.mean_squared_error(tf.zeros_like(model_outputs), model_outputs)
        gradients = gradient_tape.gradient(loss, convkb_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, convkb_model.trainable_variables))
        self.assertGreater(np.sum(
            self.default_entity_embeddings != convkb_model.embeddings_layer.entity_embeddings.numpy()
        ), 0)
        self.assertGreater(np.sum(
            self.default_relation_embeddings != convkb_model.embeddings_layer.relation_embeddings.numpy()
        ), 0)

    def test_non_trainable_embeddings(self):
        embeddings_config = EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4, trainable_embeddings=False
        )
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        convkb_model = ConvKBModel(embeddings_config, model_config, self.default_convolutions_config)
        self._set_default_weights(convkb_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        with tf.GradientTape() as gradient_tape:
            model_outputs = convkb_model(self.model_inputs, training=True)
            loss = tf.keras.losses.mean_squared_error(tf.zeros_like(model_outputs), model_outputs)
        gradients = gradient_tape.gradient(loss, convkb_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, convkb_model.trainable_variables))
        self.assertAllEqual(self.default_entity_embeddings, convkb_model.embeddings_layer.entity_embeddings.numpy())
        self.assertAllEqual(self.default_relation_embeddings, convkb_model.embeddings_layer.relation_embeddings.numpy())

    def test_trainable_convolutions(self):
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        convkb_model = ConvKBModel(self.default_embeddings_config, model_config, self.default_convolutions_config)
        self._set_default_weights(convkb_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        with tf.GradientTape() as gradient_tape:
            model_outputs = convkb_model(self.model_inputs, training=True)
            loss = tf.keras.losses.mean_squared_error(tf.zeros_like(model_outputs), model_outputs)
        gradients = gradient_tape.gradient(loss, convkb_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, convkb_model.trainable_variables))
        self.assertGreater(np.sum(self.default_kernel != convkb_model.conv_layers[0].kernel.numpy()), 0)
        self.assertGreater(np.sum(self.default_bias != convkb_model.conv_layers[0].bias.numpy()), 0)

    def test_normalize_embeddings(self):
        model_config = ConvModelConfig(include_reduce_dim_layer=False, normalize_embeddings=True)
        embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=2)
        convkb_model = ConvKBModel(embeddings_config, model_config, self.default_convolutions_config)
        convkb_model.embeddings_layer.entity_embeddings.assign([[0., 0.], [1., 0.], [0.8, 0.6]])
        convkb_model.embeddings_layer.relation_embeddings.assign([[1.6, 1.2], [2.4, 3.2]])
        convkb_model(self.model_inputs)
        convkb_model.conv_layers[0].kernel.assign(self.default_kernel)
        convkb_model.conv_layers[0].bias.assign(self.default_bias)
        outputs = convkb_model([[0, 0, 1], [1, 1, 2]], training=True)
        self.assertEqual((2, 2), outputs.shape)
        self.assertAllClose([[0.0, 0.6], [0.8, 0.2]], outputs.numpy())

    def test_dropout_layer(self):
        tf.random.set_seed(1)
        model_config = ConvModelConfig(include_reduce_dim_layer=False, dropout_rate=0.999)
        convkb_model = ConvKBModel(self.default_embeddings_config, model_config, self.default_convolutions_config)
        self._set_default_weights(convkb_model)
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose(tf.zeros_like(outputs), outputs)

    def test_gin_config(self):
        gin_config = """
            ConvModelConfig.include_reduce_dim_layer = False
            ConvModelConfig.normalize_embeddings = False
            ConvModelConfig.dropout_rate = 0.0
            ConvolutionsConfig.filter_heights = [1]
            ConvolutionsConfig.filters_count_per_height = 1
            ConvolutionsConfig.activation = "relu"
            ConvolutionsConfig.use_constant_initialization = False
            ConvKBModel.model_config = @ConvModelConfig()
            ConvKBModel.convolutions_config = @ConvolutionsConfig()
        """
        gin.parse_config(gin_config)
        embeddings_config = EmbeddingsConfig(entities_count=3, relations_count=2, embeddings_dimension=4)
        convkb_model = ConvKBModel(embeddings_config)
        self._set_default_weights(convkb_model)
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[2., 2., 1., 2.], [3., 2., 4., 3.]], outputs.numpy())


if __name__ == '__main__':
    tf.test.main()
