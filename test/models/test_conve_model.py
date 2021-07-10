import tensorflow as tf

from models.conve_model import ConvEModel, ConvEModelConfig
from layers.embeddings_layers import ObjectType, EmbeddingsConfig
from optimization.datasets import MaskedEntityOfEdgeDataset, DatasetType


class TestConvEModel(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'
    MASKED_HEAD_OBJECT_TYPES = [ObjectType.SPECIAL_TOKEN.value, ObjectType.RELATION.value, ObjectType.ENTITY.value]
    MASKED_TAIL_OBJECT_TYPES = [ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.SPECIAL_TOKEN.value]

    def setUp(self):
        dataset = MaskedEntityOfEdgeDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=5
        )
        self.model_inputs = next(iter(dataset.samples))
        self.embeddings_config = EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=6, use_special_token_embeddings=True,
        )
        self.model_config = ConvEModelConfig(
            embeddings_width=3, input_dropout_rate=0.5, conv_layer_filters=32, conv_layer_kernel_size=2,
            conv_dropout_rate=0.5, hidden_dropout_rate=0.5,
        )

    def test_model(self):
        model = ConvEModel(self.embeddings_config, self.model_config)
        outputs = model(self.model_inputs)
        self.assertAllEqual((5, 3), outputs.shape)


if __name__ == '__main__':
    tf.test.main()
