import tensorflow as tf

from conv_base_model import ConvBaseModel


class ConvKBModel(ConvBaseModel):

    def __init__(
            self, entities_count, relations_count, embedding_dimension, filter_heights, filters_count_per_height,
            include_reduce_dim_layer, pretrained_entity_embeddings=None, pretrained_relation_embeddings=None,
            trainable_embeddings=True, conv_activation='relu'
    ):
        super(ConvKBModel, self).__init__(
            entities_count, relations_count, embedding_dimension, include_reduce_dim_layer,
            pretrained_entity_embeddings, pretrained_relation_embeddings, trainable_embeddings
        )
        self._conv_layers = [
            tf.keras.layers.Conv2D(filters_count_per_height, kernel_size=(filter_height, 3), activation=conv_activation)
            for filter_height in filter_heights
        ]

    @property
    def conv_layers(self):
        return self._conv_layers

