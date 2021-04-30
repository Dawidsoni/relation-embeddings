from abc import ABC
import tensorflow as tf
import numpy as np
import os

from layers.embeddings_layers import EmbeddingsConfig, EmbeddingsExtractionLayer


class KnowledgeCompletionModel(tf.keras.Model, ABC):

    def __init__(self, embeddings_config: EmbeddingsConfig):
        super(KnowledgeCompletionModel, self).__init__()
        self.embeddings_layer = EmbeddingsExtractionLayer(embeddings_config)

    def get_trainable_variables_at_training_step(self, training_step):
        return self.trainable_variables

    def save_with_embeddings(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_weights(filepath=os.path.join(path, "saved_weights.tf"), save_format="tf")
        np.save(file=os.path.join(path, "entity_embeddings"), arr=self.embeddings_layer.entity_embeddings.numpy())
        np.save(file=os.path.join(path, "relation_embeddings"), arr=self.embeddings_layer.relation_embeddings.numpy())
        if self.embeddings_layer.config.use_special_token_embeddings:
            special_token_embeddings = self.embeddings_layer.special_token_embeddings.numpy()
            np.save(file=os.path.join(path, "special token embeddings"), arr=special_token_embeddings)
        if self.embeddings_layer.config.use_position_embeddings:
            position_embeddings = self.embeddings_layer.position_embeddings_layer.position_embeddings
            np.save(file=os.path.join(path, "position_embeddings"), arr=position_embeddings.numpy())
