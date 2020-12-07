import numpy as np
import tensorflow as tf
import gin.tf

from losses import LossObject, OptimizedMetric
from transe_model import TranseModel
from conv_base_model import DataConfig, ModelConfig


class TestTranseModel(tf.test.TestCase):

    def setUp(self):
        self.default_norm_inputs = np.array([[3.0, 4.0], [6.0, 8.0]])
        self.default_softplus_inputs = np.array([[2.0], [3.0]])
        gin.clear_config()

    def test_norm_metric_outputs(self):
        loss_object = LossObject(
            OptimizedMetric.NORM, regularization_strength=1.0, norm_metric_order=2, norm_metric_margin=1.0
        )
        losses = loss_object.get_losses_of_positive_samples(self.default_norm_inputs)
        self.assertAllClose([5., 10.], losses)

    def test_norm_metric_order(self):
        loss_object = LossObject(
            OptimizedMetric.NORM, regularization_strength=1.0, norm_metric_order=1, norm_metric_margin=1.0
        )
        losses = loss_object.get_losses_of_positive_samples(self.default_norm_inputs)
        self.assertAllClose([7., 14.], losses)

    def test_softplus_metric(self):
        loss_object = LossObject(OptimizedMetric.SOFTPLUS, regularization_strength=1.0)
        losses = loss_object.get_losses_of_positive_samples(self.default_softplus_inputs)
        self.assertAllClose([2.126928, 3.048587], losses)

    def test_mean_loss_pairs_norm_metric(self):
        loss_object = LossObject(
            OptimizedMetric.NORM, regularization_strength=1.0, norm_metric_order=2, norm_metric_margin=1.0
        )
        loss = loss_object.get_mean_loss_of_pairs(
            positive_samples=self.default_norm_inputs, negative_samples=self.default_norm_inputs[[1, 0]]
        )
        self.assertAllClose(3.0, loss)

    def test_mean_loss_pairs_norm_metric_margin(self):
        loss_object = LossObject(
            OptimizedMetric.NORM, regularization_strength=1.0, norm_metric_order=2, norm_metric_margin=3.0
        )
        loss = loss_object.get_mean_loss_of_pairs(
            positive_samples=self.default_norm_inputs, negative_samples=self.default_norm_inputs[[1, 0]]
        )
        self.assertAllClose(4.0, loss)

    def test_mean_loss_pairs_softplus_metric(self):
        loss_object = LossObject(OptimizedMetric.SOFTPLUS, regularization_strength=1.0)
        loss = loss_object.get_mean_loss_of_pairs(
            positive_samples=self.default_softplus_inputs, negative_samples=self.default_softplus_inputs[[1, 0]]
        )
        self.assertAllClose(0.668878, loss)

    def test_gin_config(self):
        gin_config = """
            LossObject.optimized_metric = %OptimizedMetric.NORM
            LossObject.regularization_strength = 1.0
            LossObject.norm_metric_order = 2
            LossObject.norm_metric_margin = 1.0
        """
        gin.parse_config(gin_config)
        loss_object = LossObject()
        losses = loss_object.get_losses_of_positive_samples(self.default_norm_inputs)
        self.assertAllClose([5., 10.], losses)

    def test_norm_metric_norm_order_undefined(self):
        loss_object = LossObject(OptimizedMetric.NORM, regularization_strength=1.0, norm_metric_margin=1.0)
        with self.assertRaises(ValueError) as error:
            loss_object.get_losses_of_positive_samples(self.default_norm_inputs)
        self.assertIn('norm_metric_order has to be defined when norm metric is used', str(error.exception))

    def test_norm_metric_norm_metric_margin_undefined(self):
        loss_object = LossObject(OptimizedMetric.NORM, regularization_strength=1.0, norm_metric_order=2)
        with self.assertRaises(ValueError) as error:
            loss_object.get_mean_loss_of_pairs(self.default_norm_inputs, self.default_norm_inputs)
        self.assertIn('norm_metric_margin has to be defined when norm metric is used', str(error.exception))

    def test_softplus_metric_invalid_shape(self):
        loss_object = LossObject(OptimizedMetric.SOFTPLUS, regularization_strength=1.0)
        with self.assertRaises(ValueError) as error:
            loss_object.get_losses_of_positive_samples(self.default_norm_inputs)
        self.assertIn('incompatible with embeddings of shape greater than 1', str(error.exception))

    def test_get_regularization_loss(self):
        pretrained_entity_embeddings = tf.ones(shape=(3, 4))
        pretrained_relations_embeddings = 2 * tf.ones(shape=(2, 4))
        data_config = DataConfig(
            entities_count=3, relations_count=2, pretrained_entity_embeddings=pretrained_entity_embeddings,
            pretrained_relations_embeddings=pretrained_relations_embeddings
        )
        model_config = ModelConfig(embeddings_dimension=4, include_reduce_dim_layer=False)
        transe_model = TranseModel(data_config, model_config)
        loss_object = LossObject(OptimizedMetric.SOFTPLUS, regularization_strength=0.1)
        self.assertAllClose(0.28, loss_object.get_regularization_loss(transe_model))


if __name__ == '__main__':
    tf.test.main()
