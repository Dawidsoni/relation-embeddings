import os
import time
import gin.tf
import string
import random
import logging

import knowledge_base_state_factory
import utils


class TrainingStopper(object):

    def __init__(self, iterations_to_stop=5):
        self.iterations_to_stop = iterations_to_stop
        self.best_value = float("-inf")
        self.iterations_since_best_value = -1

    def add_metric_value(self, metric_value):
        if metric_value > self.best_value:
            self.best_value = metric_value
            self.iterations_since_best_value = 0
        else:
            self.iterations_since_best_value += 1

    def should_training_stop(self):
        return self.iterations_since_best_value >= self.iterations_to_stop

    def last_value_optimal(self):
        return self.iterations_since_best_value == 0


def log_experiment_information(logger, experiment_name, gin_configs, gin_bindings):
    logger.info(f"Starting experiment '{experiment_name}'")
    for gin_config in gin_configs:
        with open(gin_config, mode='r') as file_stream:
            file_content = "".join(file_stream.readlines())
            log_separator = 120 * "="
            logger.info(f"Using Gin configuration: {gin_config}:\n{file_content}\n{log_separator}")
    for gin_binding in gin_bindings:
        logger.info(f"Using Gin binding: {gin_binding}")


@gin.configurable(whitelist=['threshold'])
def mrr_too_low(mrr, threshold=0.0):
    return mrr < threshold


def train_and_evaluate_model(experiment_config, experiment_id, logger):
    tensorboard_folder = os.path.join(experiment_config.tensorboard_outputs_folder, experiment_id)
    state = knowledge_base_state_factory.create_knowledge_base_state(tensorboard_folder, logger)
    training_stopper = TrainingStopper()
    training_step = 1
    for training_samples in state.training_dataset.samples:
        training_loss = state.model_trainer.train_step(training_samples, training_step)
        if training_step % 100 == 0:
            logger.info(f"Performing training step {training_step}")
            logger.info(f"Loss value: {training_loss: .3f}")
        if training_step == 1:
            logger.info(f"Parameters count of a model: {state.model.count_params()}")
        if training_step % experiment_config.steps_per_evaluation == 0:
            logger.info(f"Evaluating a model on training data")
            state.training_evaluator.evaluation_step(training_step)
            logger.info(f"Evaluating a model on validation data")
            evaluation_mrr_metric = state.validation_evaluator.evaluation_step(training_step)
            training_stopper.add_metric_value(evaluation_mrr_metric)
            if training_stopper.should_training_stop() or (
                    training_step >= 4 * experiment_config.steps_per_evaluation and mrr_too_low(evaluation_mrr_metric)):
                logger.info(f"Finishing experiment due to high value of evaluation losses: {evaluation_mrr_metric}")
                break
            elif training_stopper.last_value_optimal():
                logger.info(f"Updating the best model found (evaluation losses: {evaluation_mrr_metric}")
                state.test_evaluator.build_model()
                state.best_model.set_weights(state.model.get_weights())
        training_step += 1
        if training_step >= experiment_config.training_steps:
            break
    logger.info("Finished training a model")
    state.test_evaluator.log_metrics(logger)
    path_to_save_model = os.path.join(experiment_config.model_save_folder, experiment_id)
    state.best_model.save_with_embeddings(path_to_save_model)
    logger.info("Finished evaluating a model")
    logging.shutdown()


def prepare_and_train_model(gin_configs, gin_bindings):
    utils.init_gin_configurables()
    gin.parse_config_files_and_bindings(gin_configs, gin_bindings)
    experiment_config = utils.ExperimentConfig()
    random_id = "".join([random.choice(string.ascii_lowercase) for _ in range(4)])
    experiment_id = f"{experiment_config.experiment_name}_{random_id}{int(time.time())}"
    logger = utils.init_or_get_logger(experiment_config.logs_output_folder, experiment_id)
    log_experiment_information(logger, experiment_config.experiment_name, gin_configs, gin_bindings)
    train_and_evaluate_model(experiment_config, experiment_id, logger)


if __name__ == '__main__':
    training_args = utils.parse_gin_args()
    prepare_and_train_model(training_args.gin_configs, training_args.gin_bindings)
