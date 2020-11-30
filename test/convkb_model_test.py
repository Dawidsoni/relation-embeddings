import numpy as np
import tensorflow as tf
import gin.tf

from conv_base_model import DataConfig, ModelConfig
from convkb_model import ConvKBModel, ConvolutionsConfig


class TestConvKBModel(tf.test.TestCase):

    def _set_default_weights(self, model):
        model.entity_embeddings.assign(self.default_entity_embeddings)
        model.relation_embeddings.assign(self.default_relation_embeddings)
        model(self.model_inputs)
        model.conv_layers[0].kernel.assign(self.default_kernel)
        model.conv_layers[0].bias.assign(self.default_bias)

    def setUp(self):
        self.default_data_config = DataConfig(entities_count=3, relations_count=2)
        self.default_convolutions_config = ConvolutionsConfig(
            filter_heights=[1], filters_count_per_height=1, activation="relu"
        )
        self.default_entity_embeddings = np.array(
            [[0., 0., 0., 0.], [1., 1., 2., 1.], [2., 2., 2., 2.]], dtype=np.float32
        )
        self.default_relation_embeddings = np.array([[3., 3., 3., 3.], [4., 3., 4., 4.]], dtype=np.float32)
        self.default_kernel = np.array([[[[1.0]], [[1.0]], [[-1.0]]]])
        self.default_bias = np.array([0.0])
        self.model_inputs = np.array([[0, 0, 1], [1, 1, 2]])

    def test_outputs(self):
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False)
        convkb_model = ConvKBModel(self.default_data_config, model_config, self.default_convolutions_config)
        self._set_default_weights(convkb_model)
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[2., 2., 1., 2.], [3., 2., 4., 3.]], outputs.numpy())

    def test_activation(self):
        data_config = DataConfig(entities_count=3, relations_count=2)
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False)
        convolutions_config = ConvolutionsConfig(filter_heights=[1], filters_count_per_height=1, activation="relu")
        convkb_model = ConvKBModel(data_config, model_config, convolutions_config)
        self._set_default_weights(convkb_model)
        convkb_model.relation_embeddings.assign(-self.default_relation_embeddings)
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose(tf.zeros_like(outputs), outputs.numpy())

    def test_filter_heights(self):
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False)
        convolutions_config = ConvolutionsConfig(filter_heights=[1, 2], filters_count_per_height=1, activation="relu")
        convkb_model = ConvKBModel(self.default_data_config, model_config, convolutions_config)
        convkb_model.entity_embeddings.assign(self.default_entity_embeddings)
        convkb_model.relation_embeddings.assign(self.default_relation_embeddings)
        convkb_model(self.model_inputs)
        convkb_model.conv_layers[0].kernel.assign(self.default_kernel)
        convkb_model.conv_layers[0].bias.assign(self.default_bias)
        convkb_model.conv_layers[1].kernel.assign([[[[1.0]], [[1.0]], [[-1.0]]], [[[-1.0]], [[0.0]], [[2.0]]]])
        convkb_model.conv_layers[1].bias.assign([1.0])
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 7), outputs.shape)
        self.assertAllClose([[2., 2., 1., 2., 5., 7., 4.], [3., 2., 4., 3., 7., 5., 8.]], outputs.numpy())

    def test_filters_count_per_height(self):
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False)
        convolutions_config = ConvolutionsConfig(filter_heights=[1], filters_count_per_height=2, activation="relu")
        convkb_model = ConvKBModel(self.default_data_config, model_config, convolutions_config)
        convkb_model.entity_embeddings.assign(self.default_entity_embeddings)
        convkb_model.relation_embeddings.assign(self.default_relation_embeddings)
        convkb_model(self.model_inputs)
        convkb_model.conv_layers[0].kernel.assign([[[[1.0, 2.0]], [[1.0, -1.0]], [[-1.0, 0.0]]]])
        convkb_model.conv_layers[0].bias.assign([0.0, 1.0])
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 8), outputs.shape)
        self.assertAllClose([[2., 0., 2., 0., 1., 0., 2., 0.], [3., 0., 2., 0., 4., 1., 3., 0.]], outputs.numpy())

    def test_reduce_dim_layer(self):
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=True)
        convkb_model = ConvKBModel(self.default_data_config, model_config, self.default_convolutions_config)
        self._set_default_weights(convkb_model)
        convkb_model.reduce_layer.set_weights([np.array([[1.], [2.], [1.], [0.]]), np.array([-10.])])
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 1), outputs.shape)
        self.assertAllClose([[-3.], [1.]], outputs.numpy())

    def test_pretrained_embeddings(self):
        data_config = DataConfig(
            entities_count=3, relations_count=2, pretrained_entity_embeddings=self.default_entity_embeddings,
            pretrained_relations_embeddings=self.default_relation_embeddings
        )
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False)
        convkb_model = ConvKBModel(data_config, model_config, self.default_convolutions_config)
        self._set_default_weights(convkb_model)
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[2., 2., 1., 2.], [3., 2., 4., 3.]], outputs.numpy())

    def test_trainable_embeddings(self):
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False)
        convkb_model = ConvKBModel(self.default_data_config, model_config, self.default_convolutions_config)
        self._set_default_weights(convkb_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        with tf.GradientTape() as gradient_tape:
            model_outputs = convkb_model(self.model_inputs, training=True)
            loss = tf.keras.losses.mean_squared_error(tf.zeros_like(model_outputs), model_outputs)
        gradients = gradient_tape.gradient(loss, convkb_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, convkb_model.trainable_variables))
        self.assertGreater(np.sum(self.default_entity_embeddings != convkb_model.entity_embeddings.numpy()), 0)
        self.assertGreater(np.sum(self.default_relation_embeddings != convkb_model.relation_embeddings.numpy()), 0)

    def test_non_trainable_embeddings(self):
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False, trainable_embeddings=False)
        convkb_model = ConvKBModel(self.default_data_config, model_config, self.default_convolutions_config)
        self._set_default_weights(convkb_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        with tf.GradientTape() as gradient_tape:
            model_outputs = convkb_model(self.model_inputs, training=True)
            loss = tf.keras.losses.mean_squared_error(tf.zeros_like(model_outputs), model_outputs)
        gradients = gradient_tape.gradient(loss, convkb_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, convkb_model.trainable_variables))
        self.assertAllEqual(self.default_entity_embeddings, convkb_model.entity_embeddings.numpy())
        self.assertAllEqual(self.default_relation_embeddings, convkb_model.relation_embeddings.numpy())

    def test_trainable_convolutions(self):
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False)
        convkb_model = ConvKBModel(self.default_data_config, model_config, self.default_convolutions_config)
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
        model_config = ModelConfig(embeddings_dimension=2, include_reduce_dim_layer=False, normalize_embeddings=True)
        convkb_model = ConvKBModel(self.default_data_config, model_config, self.default_convolutions_config)
        convkb_model.entity_embeddings.assign([[0., 0.], [1., 0.], [0.8, 0.6]])
        convkb_model.relation_embeddings.assign([[1.6, 1.2], [2.4, 3.2]])
        convkb_model(self.model_inputs)
        convkb_model.conv_layers[0].kernel.assign(self.default_kernel)
        convkb_model.conv_layers[0].bias.assign(self.default_bias)
        outputs = convkb_model([[0, 0, 1], [1, 1, 2]], training=True)
        self.assertEqual((2, 2), outputs.shape)
        self.assertAllClose([[0.0, 0.6], [0.8, 0.2]], outputs.numpy())

    def test_dropout_layer(self):
        tf.random.set_seed(1)
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False, dropout_rate=0.999)
        convkb_model = ConvKBModel(self.default_data_config, model_config, self.default_convolutions_config)
        self._set_default_weights(convkb_model)
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose(tf.zeros_like(outputs), outputs)

    def test_gin_config(self):
        gin_config = """
            ModelConfig.embeddings_dimension = 4
            ModelConfig.trainable_embeddings = True
            ModelConfig.include_reduce_dim_layer = False
            ModelConfig.normalize_embeddings = False
            ModelConfig.dropout_rate = 0.0
            ConvolutionsConfig.filter_heights = [1]
            ConvolutionsConfig.filters_count_per_height = 1
            ConvolutionsConfig.activation = "relu"
            ConvKBModel.model_config = @ModelConfig()
            ConvKBModel.convolutions_config = @ConvolutionsConfig()
        """
        gin.parse_config(gin_config)
        data_config = DataConfig(entities_count=3, relations_count=2)
        convkb_model = ConvKBModel(data_config)
        self._set_default_weights(convkb_model)
        outputs = convkb_model(self.model_inputs, training=True)
        self.assertEqual((2, 4), outputs.shape)
        self.assertAllClose([[2., 2., 1., 2.], [3., 2., 4., 3.]], outputs.numpy())


if __name__ == '__main__':
    tf.test.main()
