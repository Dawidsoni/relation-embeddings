import os
import gin.tf
import time

import knowledge_base_state_factory
import utils


@gin.configurable
def load_and_set_model_weights(state, model_weights_path=gin.REQUIRED):
    state.model.load_weights(model_weights_path)
    state.best_model.load_weights(model_weights_path)


def prepare_and_evaluate_model(gin_configs, gin_bindings):
    utils.init_gin_configurables()
    gin.parse_config_files_and_bindings(gin_configs, gin_bindings)
    experiment_config = utils.ExperimentConfig()
    experiment_id = f"evaluation_{experiment_config.experiment_name}_{int(time.time())}"
    logger = utils.init_or_get_logger(experiment_config.logs_output_folder, experiment_id)
    tensorboard_folder = os.path.join(experiment_config.tensorboard_outputs_folder, experiment_id)
    state = knowledge_base_state_factory.create_knowledge_base_state(tensorboard_folder)
    load_and_set_model_weights(state)
    state = knowledge_base_state_factory.create_knowledge_base_state(tensorboard_folder)
    state.test_evaluator.log_metrics(logger)


if __name__ == '__main__':
    evaluation_args = utils.parse_gin_args()
    prepare_and_evaluate_model(evaluation_args.gin_configs, evaluation_args.gin_bindings)
