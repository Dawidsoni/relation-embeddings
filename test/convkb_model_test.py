import numpy as np
import tensorflow as tf

from convkb_model import ConvKBModel


class TestConvKBModel(tf.test.TestCase):

    def test_pretrained_embeddings(self):
        entity_embeddings = np.array([[0., 0., 0., 0.], [1., 1., 1., 1.], [2., 2., 2., 2.]], dtype=np.float32)
        relation_embeddings = np.array([[3., 3., 3., 3.], [4., 4., 4., 4.]], dtype=np.float32)
        model = ConvKBModel(
            entities_count=3, relations_count=2, embedding_dimension=4, filter_heights=[1], filters_count_per_height=1,
            pretrained_entity_embeddings=entity_embeddings, pretrained_relation_embeddings=relation_embeddings,
            include_reduce_dim_layer=False
        )
        self.assertAllClose(entity_embeddings, model.entity_embeddings.numpy())
        self.assertAllClose(relation_embeddings, model.relation_embeddings.numpy())

    def test_model_one_filter_output(self):
        entity_embeddings = np.array([[-1., 0., 0., -1.], [1., 1., 1., 1.], [2., 2., 2., 2.]], dtype=np.float32)
        relation_embeddings = np.array([[3., 3., 3., 3.], [4., 4., 4., 4.]], dtype=np.float32)
        model = ConvKBModel(
            entities_count=3, relations_count=2, embedding_dimension=4, filter_heights=[1], filters_count_per_height=1,
            pretrained_entity_embeddings=entity_embeddings, pretrained_relation_embeddings=relation_embeddings,
            include_reduce_dim_layer=False
        )
        model(np.array([[0, 0, 1], [1, 1, 2]]))
        model.conv_layers[0].set_weights([np.array([[[[1.]], [[1.]], [[-1.]]]]), np.array([-1.])])
        output = model(np.array([[0, 0, 1], [1, 1, 2]]))
        self.assertEqual((2, 4), output.shape)
        self.assertAllClose([[0., 1., 1., 0.], [2., 2., 2., 2.]], output.numpy())

    def test_model_multiple_filters_output(self):
        entity_embeddings = np.array([[-1., 0., 0., -1.], [1., 1., 1., 1.], [2., 2., 2., 2.]], dtype=np.float32)
        relation_embeddings = np.array([[3., 3., 3., 3.], [4., 4., 4., 4.]], dtype=np.float32)
        model = ConvKBModel(
            entities_count=3, relations_count=2, embedding_dimension=4, filter_heights=[1, 2],
            filters_count_per_height=3, pretrained_entity_embeddings=entity_embeddings,
            pretrained_relation_embeddings=relation_embeddings, include_reduce_dim_layer=False
        )
        output = model(np.array([[0, 0, 1], [1, 1, 2]]))
        self.assertEqual((2, 21), output.shape)

    def test_model_reduce(self):
        entity_embeddings = np.array([[-1., 0., 0., -1.], [1., 1., 1., 1.], [2., 2., 2., 2.]], dtype=np.float32)
        relation_embeddings = np.array([[3., 3., 3., 3.], [4., 4., 4., 4.]], dtype=np.float32)
        model = ConvKBModel(
            entities_count=3, relations_count=2, embedding_dimension=4, filter_heights=[1], filters_count_per_height=1,
            pretrained_entity_embeddings=entity_embeddings, pretrained_relation_embeddings=relation_embeddings,
            include_reduce_dim_layer=True
        )
        model(np.array([[0, 0, 1], [1, 1, 2]]))
        model.conv_layers[0].set_weights([np.array([[[[1.]], [[1.]], [[-1.]]]]), np.array([-1.])])
        model.reduce_layer.set_weights([np.array([[1.], [1.], [1.], [1.]]), np.array([-5.])])
        output = model(np.array([[0, 0, 1], [1, 1, 2]]))
        self.assertEqual((2, 1), output.shape)
        self.assertAllClose([[-3.], [3.]], output.numpy())


if __name__ == '__main__':
    tf.test.main()
