import tensorflow as tf
import gin.tf

from models.hierarchical_softmax_model import HierarchicalTransformerModel, HierarchicalTransformerModelConfig
from layers.embeddings_layers import EmbeddingsConfig
from datasets.softmax_datasets import CombinedMaskedDataset, InputNeighboursDataset, OutputNeighboursDataset, \
    CombinedMaskedDatasetTrainingMode
from datasets.dataset_utils import DatasetType
from models.transformer_softmax_model import TransformerSoftmaxModel, TransformerSoftmaxModelConfig


class TestHierarchicalTransformerModel(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def setUp(self):
        gin.enter_interactive_mode()
        gin.parse_config("""
            InputNeighboursDataset.dataset_id = "input_neighbours"
            InputNeighboursDataset.max_neighbours_count = 1
            InputNeighboursDataset.mask_source_entity_pbty = 0.5
            OutputNeighboursDataset.dataset_id = "output_neighbours"
            OutputNeighboursDataset.max_neighbours_count = 2
            OutputNeighboursDataset.mask_source_entity_pbty = 0.25
            
            StackedTransformerEncodersLayer.layers_count = 3
            StackedTransformerEncodersLayer.attention_heads_count = 4
            StackedTransformerEncodersLayer.attention_head_dimension = 5
            StackedTransformerEncodersLayer.pointwise_hidden_layer_dimension = 4
            StackedTransformerEncodersLayer.dropout_rate = 0.5
            StackedTransformerEncodersLayer.share_encoder_parameters = False
            StackedTransformerEncodersLayer.share_encoder_parameters = False
            StackedTransformerEncodersLayer.encoder_layer_type = %TransformerEncoderLayerType.PRE_LAYER_NORM
        """)
        template1 = InputNeighboursDataset
        template2 = OutputNeighboursDataset
        independent_losses_dataset = CombinedMaskedDataset(
            dataset_templates=[template1, template2], dataset_type=DatasetType.TRAINING,
            data_directory=self.DATASET_PATH, shuffle_dataset=True, batch_size=4, inference_mode=False,
            dataset_id="independent_losses_dataset", training_mode=CombinedMaskedDatasetTrainingMode.INDEPENDENT_LOSSES,
        )
        self.independent_losses_samples = next(iter(independent_losses_dataset.samples))
        joint_loss_dataset = CombinedMaskedDataset(
            dataset_templates=[template1, template2], dataset_type=DatasetType.TRAINING,
            data_directory=self.DATASET_PATH, shuffle_dataset=True, batch_size=4, inference_mode=False,
            dataset_id="joint_loss_dataset", training_mode=CombinedMaskedDatasetTrainingMode.JOINT_LOSS,
        )
        self.joint_loss_samples = next(iter(joint_loss_dataset.samples))
        embeddings_config = EmbeddingsConfig(
            entities_count=3, relations_count=2, embeddings_dimension=6, use_special_token_embeddings=True,
        )
        transformer_config = TransformerSoftmaxModelConfig(
            use_pre_normalization=True, pre_dropout_rate=0.5
        )
        self.submodel1 = TransformerSoftmaxModel(embeddings_config, transformer_config)
        self.submodel2 = TransformerSoftmaxModel(embeddings_config, transformer_config)
        self.default_model_config = HierarchicalTransformerModelConfig(dropout_rate=0.5, layers_count=2)

    def test_model_on_independent_losses_samples(self):
        model = HierarchicalTransformerModel(
            ids_to_models={"input_neighbours": self.submodel1, "output_neighbours": self.submodel2},
            config=self.default_model_config,
        )
        self.assertAllEqual((2, 4, 3), model(self.independent_losses_samples).shape)

    def test_model_on_joint_loss_samples(self):
        model = HierarchicalTransformerModel(
            ids_to_models={"input_neighbours": self.submodel1, "output_neighbours": self.submodel2},
            config=self.default_model_config,
        )
        self.assertAllEqual((4, 3), model(self.joint_loss_samples).shape)


if __name__ == '__main__':
    tf.test.main()
