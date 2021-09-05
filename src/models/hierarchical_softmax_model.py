import dataclasses
import collections
import gin.tf
import tensorflow as tf

from datasets.softmax_datasets import CombinedMaskedDatasetTrainingMode
from layers.transformer_layers import StackedTransformerEncodersLayer
from models.knowledge_completion_model import KnowledgeCompletionModel
from optimization import parameters_factory


@gin.configurable
@dataclasses.dataclass
class HierarchicalTransformerModelConfig(object):
    dropout_rate: float
    layers_count: int


@gin.configurable
class HierarchicalTransformerModel(KnowledgeCompletionModel):

    def __init__(self, ids_to_models, config: HierarchicalTransformerModelConfig = gin.REQUIRED):
        super().__init__()
        self.ids_to_models = collections.OrderedDict(ids_to_models)
        self.config = config
        self._validate_submodels()
        self.embeddings_dimension = list(self.ids_to_models.values())[0].embeddings_layer.config.embeddings_dimension
        self.entities_count = list(self.ids_to_models.values())[0].embeddings_layer.config.entities_count
        self.dropout_layer = tf.keras.layers.Dropout(rate=self.config.dropout_rate)
        self.transformer_layer = StackedTransformerEncodersLayer(layers_count=self.config.layers_count)
        self.projection_layer = tf.keras.layers.Dense(
            units=self.entities_count,
            activation=parameters_factory.get_activation(),
            kernel_initializer=parameters_factory.get_parameters_initializer(),
        )

    def _validate_submodels(self):
        embeddings_dimension = None
        for model in self.ids_to_models.values():
            if embeddings_dimension is None:
                embeddings_dimension = model.embeddings_layer.config.embeddings_dimension
            if embeddings_dimension != model.embeddings_layer.config.embeddings_dimension:
                raise ValueError(
                    "Expected all models to contain embeddings of the same dimension, got ", embeddings_dimension,
                    " != ", model.embeddings_layer.config.embeddings_dimension
                )

    def _get_model_inputs(self, id_of_model, inputs):
        return {
            input_name.split("@")[1]: input_values for input_name, input_values in inputs.items()
            if input_name.startswith(id_of_model)
        }

    def _get_independent_losses_outputs(self, inputs):
        outputs = []
        for id_of_model, model in self.ids_to_models.items():
            outputs.append(model(self._get_model_inputs(id_of_model, inputs), use_projection_layer=True))
        return tf.stack(outputs)

    def _project_with_submodels(self, list_of_inputs):
        list_of_outputs = []
        for model, inputs in zip(self.ids_to_models.values(), list_of_inputs):
            outputs = tf.linalg.matmul(inputs, model.get_similarity_matrix(), transpose_b=True)
            outputs += model.projection_bias
            list_of_outputs.append(outputs)
        return list_of_outputs

    def _get_joint_loss_outputs(self, inputs):
        outputs = []
        for id_of_model, model in self.ids_to_models.items():
            outputs.append(model(self._get_model_inputs(id_of_model, inputs), use_projection_layer=False))
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
        outputs = self.dropout_layer(outputs)
        outputs = self.transformer_layer(outputs)
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
        outputs = tf.unstack(outputs)
        outputs = self._project_with_submodels(outputs)
        outputs = tf.reduce_sum(outputs, axis=0)
        return outputs

    def call(self, inputs, training=None, **kwargs):
        if inputs["mode"][0] == CombinedMaskedDatasetTrainingMode.INDEPENDENT_LOSSES.value:
            return self._get_independent_losses_outputs(inputs)
        elif inputs["mode"][0] == CombinedMaskedDatasetTrainingMode.JOINT_LOSS.value:
            return self._get_joint_loss_outputs(inputs)
        else:
            raise ValueError(f"Invalid mode: {inputs['mode'][0]}")

    def get_config(self):
        config = {}
        for model_id, model in self.ids_to_models.items():
            config.update({
                f"{model_id}@{p_name}": p_value
                for p_name, p_value in dataclasses.asdict(model.model_config).items()
            })
        return config
