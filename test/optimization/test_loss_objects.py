import numpy as np
import tensorflow as tf
import gin.tf

from optimization.loss_objects import NormLossObject, SoftplusLossObject, CrossEntropyLossObject, \
    BinaryCrossEntropyLossObject
from models.transe_model import TranseModel
from models.conv_base_model import EmbeddingsConfig, ConvModelConfig


class TestLossObjects(tf.test.TestCase):

    def setUp(self):
        self.default_norm_inputs = np.array([[3.0, 4.0], [6.0, 8.0]])
        self.default_softplus_inputs = np.array([[2.0], [3.0]])
        gin.clear_config()
        gin.parse_config("""
            LossObject.regularization_strength = 0.1
        """)

    def test_norm_metric_outputs(self):
        loss_object = NormLossObject(order=2, margin=1.0)
        losses = loss_object.get_losses_of_positive_samples(self.default_norm_inputs)
        self.assertAllClose([5., 10.], losses)

    def test_norm_metric_order(self):
        loss_object = NormLossObject(order=1, margin=1.0)
        losses = loss_object.get_losses_of_positive_samples(self.default_norm_inputs)
        self.assertAllClose([7., 14.], losses)

    def test_softplus_metric(self):
        loss_object = SoftplusLossObject(regularization_strength=1.0)
        losses = loss_object.get_losses_of_positive_samples(self.default_softplus_inputs)
        self.assertAllClose([2.126928, 3.048587], losses)

    def test_mean_loss_pairs_norm_metric(self):
        loss_object = NormLossObject(order=2, margin=1.0)
        loss = loss_object.get_mean_loss_of_pairs(
            positive_samples=self.default_norm_inputs, negative_samples=self.default_norm_inputs[[1, 0]]
        )
        self.assertAllClose(3.0, loss)

    def test_mean_loss_pairs_norm_metric_margin(self):
        loss_object = NormLossObject(order=2, margin=3.0)
        loss = loss_object.get_mean_loss_of_pairs(
            positive_samples=self.default_norm_inputs, negative_samples=self.default_norm_inputs[[1, 0]]
        )
        self.assertAllClose(4.0, loss)

    def test_mean_loss_pairs_softplus_metric(self):
        loss_object = SoftplusLossObject()
        loss = loss_object.get_mean_loss_of_pairs(
            positive_samples=self.default_softplus_inputs, negative_samples=self.default_softplus_inputs[[1, 0]]
        )
        self.assertAllClose(0.668878, loss)

    def test_softplus_metric_invalid_shape(self):
        loss_object = SoftplusLossObject()
        with self.assertRaises(ValueError) as error:
            loss_object.get_losses_of_positive_samples(self.default_norm_inputs)
        self.assertIn('incompatible with embeddings of shape greater than 1', str(error.exception))

    def test_get_regularization_loss(self):
        pretrained_entity_embeddings = tf.ones(shape=(3, 4))
        pretrained_relation_embeddings = 2 * tf.ones(shape=(2, 4))
        embeddings_config = EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4,
            pretrained_entity_embeddings=pretrained_entity_embeddings,
            pretrained_relation_embeddings=pretrained_relation_embeddings
        )
        model_config = ConvModelConfig(include_reduce_dim_layer=False)
        transe_model = TranseModel(embeddings_config, model_config)
        loss_object = SoftplusLossObject(regularization_strength=0.1)
        self.assertAllClose(0.28, loss_object.get_regularization_loss(transe_model))

    def test_binary_cross_entropy_loss_object_positive_losses(self):
        loss_object = BinaryCrossEntropyLossObject(label_smoothing=0.0)
        self.assertAllClose(
            [0.006, 0.693, 5.007],
            loss_object.get_losses_of_positive_samples(samples=np.array([[5.0], [0.0], [-5.0]], dtype=np.float32)),
            atol=1e-2,
        )

    def test_binary_cross_entropy_loss_object_pairs_losses(self):
        loss_object = BinaryCrossEntropyLossObject(label_smoothing=0.0)
        pairs_losses = loss_object.get_losses_of_pairs(
            positive_samples=np.array([[5.0], [0.0], [-5.0]], dtype=np.float32),
            negative_samples=np.array([[5.0], [0.0], [-5.0]], dtype=np.float32),
        )
        self.assertAllClose([2.507, 0.693, 2.507], pairs_losses, atol=1e-2)

    def test_binary_cross_entropy_loss_object_label_smoothing(self):
        loss_object = BinaryCrossEntropyLossObject(label_smoothing=0.1)
        self.assertAllClose(
            [0.257, 0.693, 4.757],
            loss_object.get_losses_of_positive_samples(samples=np.array([[5.0], [0.0], [-5.0]], dtype=np.float32)),
            atol=1e-2,
        )

    def test_cross_entropy_loss_object_predicted_loss(self):
        true_labels = np.array([1, 2], dtype=np.float32)
        predictions = np.array([[0.1, 0.9, 0], [0.1, 0.8, 0.1]], dtype=np.float32)
        loss_object = CrossEntropyLossObject(label_smoothing=0.0)
        self.assertAllClose(
            1.004048, loss_object.get_mean_loss_of_samples(true_labels, predictions), atol=1e-3
        )

    def test_cross_entropy_loss_object_label_smoothing(self):
        true_labels = np.array([1, 2], dtype=np.float32)
        predictions = np.array([[0.1, 0.9, 0], [0.1, 0.8, 0.1]], dtype=np.float32)
        loss_object = CrossEntropyLossObject(label_smoothing=0.1)
        self.assertAllClose(
            1.020715, loss_object.get_mean_loss_of_samples(true_labels, predictions), atol=1e-3
        )

    def test_cross_entropy_loss_multiple_losses(self):
        true_labels = np.array([1, 2], dtype=np.float32)
        predictions = np.array([[[0.1, 0.9, 0], [0.1, 0.8, 0.1]], [[0.1, 0.9, 0], [0.1, 0.8, 0.1]]], dtype=np.float32)
        loss_object = CrossEntropyLossObject(label_smoothing=0.1)
        self.assertAllClose(
            2.04143, loss_object.get_mean_loss_of_samples(true_labels, predictions), atol=1e-3
        )

    def test_cross_entropy_losses_of_samples(self):
        true_labels = np.array([1, 2], dtype=np.float32)
        predictions = np.array([[0.1, 0.9, 0], [0.1, 0.8, 0.1]], dtype=np.float32)
        loss_object = CrossEntropyLossObject(label_smoothing=0.1)
        self.assertAllClose(
            np.array([0.675036, 1.366393], dtype=np.float32),
            loss_object.get_losses_of_samples(true_labels, predictions),
            atol=1e-3,
        )

    def test_cross_entropy_invalid_label(self):
        true_labels = np.array([1, 3], dtype=np.float32)
        predictions = np.array([[0.1, 0.9, 0], [0.1, 0.8, 0.1]], dtype=np.float32)
        loss_object = CrossEntropyLossObject(label_smoothing=0.1)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            loss_object.get_losses_of_samples(true_labels, predictions)


if __name__ == '__main__':
    tf.test.main()
