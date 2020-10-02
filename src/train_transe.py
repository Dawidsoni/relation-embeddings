import argparse
import tensorflow as tf

from dataset import Dataset
from losses import LossObject, OptimizedMetric
from train_lib import TrainLib
from eval_lib import EvalLib
from transe_model import TranseModel


def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', type=str, required=True)
    parser.add_argument('--tensorboard_directory', type=str, required=True)
    parser.add_argument('--embeddings_dimension', type=int, required=True)
    parser.add_argument('--margin_of_error', type=float, required=True)
    parser.add_argument('--loss_norm', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--optimizer_decay_steps', type=int, required=True)
    parser.add_argument('--optimizer_decay_rate', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    return parser.parse_args()


def train_transe():
    training_args = parse_training_args()
    training_dataset = Dataset(
        training_args.data_directory, graph_edges_filename='train.txt', batch_size=training_args.batch_size,
    )
    validation_dataset = Dataset(training_args.data_directory, graph_edges_filename='valid.txt', repeat_samples=True)
    model = TranseModel(
        entities_count=len(training_dataset.entity_ids),
        relations_count=len(training_dataset.relation_ids),
        embedding_dimension=training_args.embeddings_dimension,
        include_reduce_dim_layer=False
    )
    loss_object = LossObject(
        OptimizedMetric.NORM, norm_metric_order=1, norm_metric_margin=training_args.margin_of_error
    )
    learning_weight_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        training_args.learning_rate, training_args.optimizer_decay_steps, training_args.optimizer_decay_steps
    )
    train_lib = TrainLib(model, loss_object, learning_weight_schedule, regularization_strength=1.0)
    eval_lib = EvalLib(model, loss_object, validation_dataset)
    for i in range(training_args.epochs):
        for positive_inputs, negative_inputs in training_dataset.pairs_of_samples:
            train_lib.train_step(positive_inputs, negative_inputs)
        eval_lib.evaluation_step()


if __name__ == '__main__':
    train_transe()
