import numpy as np
import tensorflow as tf

from transe_model import TranseModel


class TestTranseModel(tf.test.TestCase):

    def test_model_output(self):
        model = TranseModel(entities_count=3, relations_count=2, embedding_dimension=4, include_reduce_dim_layer=False)
        model.entity_embeddings.assign(
            np.array([[0., 0., 0., 0.], [1., 1., 1., 1.], [2., 2., 2., 2.]], dtype=np.float32)
        )
        model.relation_embeddings.assign(np.array([[3., 3., 3., 3.], [4., 4., 4., 4.]], dtype=np.float32))
        output = model(np.array([[0, 0, 1], [1, 1, 2]]))
        self.assertEqual((2, 4), output.shape)
        self.assertAllClose([[2., 2., 2., 2.], [3., 3., 3., 3.]], output.numpy())

    def test_model_reduce(self):
        model = TranseModel(entities_count=3, relations_count=2, embedding_dimension=4, include_reduce_dim_layer=True)
        model.entity_embeddings.assign(
            np.array([[0., 0., 0., 0.], [1., 1., 1., 1.], [2., 2., 2., 2.]], dtype=np.float32)
        )
        model.relation_embeddings.assign(np.array([[3., 3., 3., 3.], [4., 4., 4., 4.]], dtype=np.float32))
        model(np.array([[0, 0, 1], [1, 1, 2]]))
        model.reduce_layer.set_weights([np.array([[1.], [1.], [1.], [1.]]), np.array([-10.])])
        output = model(np.array([[0, 0, 1], [1, 1, 2]]))
        self.assertEqual((2, 1), output.shape)
        self.assertAllClose([[-2.], [2.]], output.numpy())


if __name__ == '__main__':
    tf.test.main()
