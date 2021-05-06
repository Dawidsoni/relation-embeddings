import dataclasses
import gin.tf

from layers.embeddings_layers import EmbeddingsConfig
from models.conv_base_model import KnowledgeCompletionModel


@gin.configurable(blacklist=['embeddings_config'])
class TransformerConveModel(KnowledgeCompletionModel):

    def __init__(self, embeddings_config: EmbeddingsConfig):
        super().__init__(embeddings_config)

    def get_config(self):
        return {
            **dataclasses.asdict(self.embeddings_config),
            **dataclasses.asdict(self.model_config),
        }

    def call(self, inputs, training=None, **kwargs):
        pass  # TODO
