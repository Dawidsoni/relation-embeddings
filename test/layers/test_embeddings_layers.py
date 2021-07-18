import tensorflow as tf
import numpy as np

from layers.embeddings_layers import PositionEmbeddingsLayer, EmbeddingsExtractionLayer, EmbeddingsConfig, ObjectType


class TestEmbeddingsLayers(tf.test.TestCase):

    def test_position_embeddings_outputs(self):
        layer = PositionEmbeddingsLayer(
            max_inputs_length=3, initial_values=None, use_fourier_series=False, trainable=True
        )
        embeddings1 = layer(tf.ones(shape=(3, 2, 10)))
        embeddings2 = layer(tf.ones(shape=(2, 3, 10)))
        self.assertEqual(embeddings1.shape, (3, 2, 10))
        self.assertEqual(embeddings2.shape, (2, 3, 10))
        self.assertAllEqual(embeddings1[0], embeddings1[1])
        self.assertAllEqual(embeddings1[0], embeddings1[2])
        self.assertAllEqual(embeddings2[0], embeddings2[1])
        self.assertAllEqual(embeddings1[:2, :, :], embeddings2[:, :2, :])

    def test_position_embeddings_initial_values(self):
        initial_values = np.array([[2.0, 3.0, 1.0], [3.0, 4.0, 5.0]])
        layer = PositionEmbeddingsLayer(
            max_inputs_length=3, initial_values=initial_values, use_fourier_series=False, trainable=True
        )
        layer.build(input_shape=(5, 3))
        self.assertAllEqual(initial_values, layer.position_embeddings)

    def test_position_embeddings_fourier_series(self):
        layer = PositionEmbeddingsLayer(
            max_inputs_length=3, initial_values=None, use_fourier_series=True, trainable=True
        )
        layer.build(input_shape=(5, 3))
        expected_embeddings = np.array([
            [0., 1., 0.], [0.84147098, 0.54030231, 0.00215443], [0.90929743, -0.41614684, 0.00430886],
        ])
        self.assertAllClose(expected_embeddings, layer.position_embeddings)

    def test_position_embeddings_trainable(self):
        layer = PositionEmbeddingsLayer(
            max_inputs_length=3, initial_values=None, use_fourier_series=False, trainable=True
        )
        layer.build(input_shape=(5, 3))
        self.assertEqual(True, layer.position_embeddings.trainable)

    def test_position_embeddings_non_trainable(self):
        layer = PositionEmbeddingsLayer(
            max_inputs_length=3, initial_values=None, use_fourier_series=False, trainable=False
        )
        layer.build(input_shape=(5, 3))
        self.assertEqual(False, layer.position_embeddings.trainable)

    def test_embeddings_extraction_layer_outputs(self):
        layer = EmbeddingsExtractionLayer(EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4, use_special_token_embeddings=True
        ))
        inputs = {
            "object_ids": tf.constant([[1, 0], [2, 0]], dtype=tf.int32),
            "object_types": tf.constant(
                [[ObjectType.RELATION.value, ObjectType.ENTITY.value],
                 [ObjectType.ENTITY.value, ObjectType.SPECIAL_TOKEN.value]], dtype=tf.int32)
        }
        outputs = layer(inputs)
        self.assertAllEqual(layer.relation_embeddings[1], outputs[0, 0])
        self.assertAllEqual(layer.entity_embeddings[0], outputs[0, 1])
        self.assertAllEqual(layer.entity_embeddings[2], outputs[1, 0])
        self.assertAllEqual(layer.special_token_embeddings[0], outputs[1, 1])

    def test_embeddings_extraction_layer_trainable(self):
        layer = EmbeddingsExtractionLayer(EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4, trainable_embeddings=True
        ))
        self.assertTrue(layer.entity_embeddings.trainable)
        self.assertTrue(layer.relation_embeddings.trainable)

    def test_embeddings_extraction_layer_non_trainable(self):
        layer = EmbeddingsExtractionLayer(EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4, trainable_embeddings=False
        ))
        self.assertFalse(layer.entity_embeddings.trainable)
        self.assertFalse(layer.relation_embeddings.trainable)

    def test_embeddings_extraction_layer_pretrained_entity_embeddings(self):
        initial_values = np.array([[2.0, 3.0, 1.0], [3.0, 4.0, 5.0], [2.0, 5.0, 1.0]])
        layer = EmbeddingsExtractionLayer(EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4, pretrained_entity_embeddings=initial_values
        ))
        self.assertAllEqual(initial_values, layer.entity_embeddings)

    def test_embeddings_extraction_layer_pretrained_relation_embeddings(self):
        initial_values = np.array([[2.0, 3.0, 1.0], [3.0, 4.0, 5.0]])
        layer = EmbeddingsExtractionLayer(EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4, pretrained_relation_embeddings=initial_values
        ))
        self.assertAllEqual(initial_values, layer.relation_embeddings)

    def test_embeddings_extraction_layer_pretrained_special_token_embeddings(self):
        initial_values = np.array([[2.0, 3.0, 1.0]])
        layer = EmbeddingsExtractionLayer(EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4,
            use_special_token_embeddings=True, pretrained_special_token_embeddings=initial_values
        ))
        self.assertAllEqual(initial_values, layer.special_token_embeddings)

    def test_embeddings_extraction_layer_use_position_embeddings(self):
        layer = EmbeddingsExtractionLayer(EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4,
            use_special_token_embeddings=True, use_position_embeddings=True,
        ))
        inputs = {
            "object_ids": tf.constant([[1, 0], [2, 0]], dtype=tf.int32),
            "object_types": tf.constant(
                [[ObjectType.RELATION.value, ObjectType.ENTITY.value],
                 [ObjectType.ENTITY.value, ObjectType.SPECIAL_TOKEN.value]], dtype=tf.int32)
        }
        outputs = layer(inputs)
        position_embeddings = layer.position_embeddings_layer.position_embeddings
        self.assertAllEqual(layer.relation_embeddings[1] + position_embeddings[0], outputs[0, 0])
        self.assertAllEqual(layer.entity_embeddings[0] + position_embeddings[1], outputs[0, 1])
        self.assertAllEqual(layer.entity_embeddings[2] + position_embeddings[0], outputs[1, 0])
        self.assertAllEqual(layer.special_token_embeddings[0] + position_embeddings[1], outputs[1, 1])

    def test_embeddings_extraction_layer_custom_position_embeddings(self):
        layer = EmbeddingsExtractionLayer(EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4,
            use_special_token_embeddings=True, use_position_embeddings=True,
        ))
        inputs = {
            "object_ids": tf.constant([[1, 0], [2, 0]], dtype=tf.int32),
            "object_types": tf.constant(
                [[ObjectType.RELATION.value, ObjectType.ENTITY.value],
                 [ObjectType.ENTITY.value, ObjectType.SPECIAL_TOKEN.value]], dtype=tf.int32),
            "positions": tf.constant([[0, 2], [1, 0]], dtype=tf.int32),
        }
        outputs = layer(inputs)
        position_embeddings = layer.position_embeddings_layer.position_embeddings
        self.assertAllEqual(layer.relation_embeddings[1] + position_embeddings[0], outputs[0, 0])
        self.assertAllEqual(layer.entity_embeddings[0] + position_embeddings[2], outputs[0, 1])
        self.assertAllEqual(layer.entity_embeddings[2] + position_embeddings[1], outputs[1, 0])
        self.assertAllEqual(layer.special_token_embeddings[0] + position_embeddings[0], outputs[1, 1])

    def test_embeddings_extraction_layer_special_token_embeddings_used(self):
        layer = EmbeddingsExtractionLayer(EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4, use_special_token_embeddings=True,
            special_tokens_count=3,
        ))
        self.assertEqual((3, 4), layer.special_token_embeddings.shape)

    def test_embeddings_extraction_layer_special_token_embeddings_not_used(self):
        layer = EmbeddingsExtractionLayer(EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4, use_special_token_embeddings=False
        ))
        self.assertEqual((0, 4), layer.special_token_embeddings.shape)

    def test_embeddings_extraction_layer_position_embeddings_trainable(self):
        layer = EmbeddingsExtractionLayer(EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4,
            use_position_embeddings=True, position_embeddings_trainable=True
        ))
        self.assertTrue(layer.position_embeddings_layer.trainable)

    def test_embeddings_extraction_layer_position_embeddings_non_trainable(self):
        layer = EmbeddingsExtractionLayer(EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4,
            use_position_embeddings=True, position_embeddings_trainable=False
        ))
        self.assertFalse(layer.position_embeddings_layer.trainable)

    def test_embeddings_extraction_layer_pretrained_position_embeddings(self):
        initial_values = np.array([[2.0, 3.0, 1.0], [3.0, 4.0, 5.0], [2.0, 5.0, 1.0]])
        layer = EmbeddingsExtractionLayer(EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4,
            use_position_embeddings=True, pretrained_position_embeddings=initial_values,
        ))
        layer.position_embeddings_layer.build(input_shape=(3, 3))
        self.assertAllEqual(initial_values, layer.position_embeddings_layer.position_embeddings)

    def test_embeddings_extraction_layer_position_embeddings_fourier_transform(self):
        layer = EmbeddingsExtractionLayer(EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=4,
            use_position_embeddings=True, use_fourier_series_in_position_embeddings=True,
        ))
        self.assertTrue(layer.position_embeddings_layer.use_fourier_series)