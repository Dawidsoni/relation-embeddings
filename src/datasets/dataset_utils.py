import collections
import functools
import enum
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import itertools

from layers.embeddings_layers import ObjectType


TRAINING_DATASET_FILENAME = "train.txt"
VALIDATION_DATASET_FILENAME = "valid.txt"
TEST_DATASET_FILENAME = "test.txt"
ENTITIES_IDS_FILENAME = "entity2id.txt"
RELATIONS_IDS_FILENAME = "relation2id.txt"


MASKED_ENTITY_TOKEN_ID = 0
MISSING_ENTITY_TOKEN_ID = 1
MISSING_RELATION_TOKEN_ID = 2
MISSING_SOURCE_ENTITY_ID = 3
MISSING_CONTEXT_ENTITY_ID = 4
MISSING_CONTEXT_RELATION_ID = 5
EDGE_OBJECT_TYPES = (ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value)


class DatasetType(enum.Enum):
    TRAINING = 0
    VALIDATION = 1
    TEST = 2


def extract_entity_ids(data_directory):
    entities_df = pd.read_table(os.path.join(data_directory, ENTITIES_IDS_FILENAME), header=None)
    return dict(zip(entities_df[0], entities_df[1]))


def extract_relation_ids(data_directory):
    relations_df = pd.read_table(os.path.join(data_directory, RELATIONS_IDS_FILENAME), header=None)
    return dict(zip(relations_df[0], relations_df[1]))


def extract_edges_from_file(entity_ids, relation_ids, data_directory, dataset_type):
    dataset_types_filenames = {
        DatasetType.TRAINING: TRAINING_DATASET_FILENAME,
        DatasetType.VALIDATION: VALIDATION_DATASET_FILENAME,
        DatasetType.TEST: TEST_DATASET_FILENAME,
    }
    if dataset_type not in dataset_types_filenames:
        raise ValueError(f"Expected an instance of DatasetType, got {dataset_type}")
    graph_df = pd.read_table(os.path.join(data_directory, dataset_types_filenames[dataset_type]), header=None)
    return list(zip(
        [entity_ids[x] for x in graph_df[0]],
        [relation_ids[x] for x in graph_df[1]],
        [entity_ids[x] for x in graph_df[2]]
    ))


def get_int_random_variables_iterator(low, high, batch_size=100_000):
    while True:
        for random_variable in np.random.randint(low, high, size=batch_size):
            yield random_variable


def map_list_of_dicts_to_dict(list_of_dicts):
    output_dict = collections.defaultdict(list)
    for item_dict in list_of_dicts:
        for key, value in item_dict.items():
            output_dict[key].append(value)
    return output_dict


def interleave_datasets(dataset1, dataset2):
    return tf.data.Dataset.zip((dataset1, dataset2)).flat_map(
        lambda x1, x2: tf.data.Dataset.from_tensors(x1).concatenate(tf.data.Dataset.from_tensors(x2))
    )


def interleave_datasets_with_probs(datasets, probs, signature):
    def generator():
        iterators = [iter(dataset.repeat()) for dataset in datasets]
        while True:
            index = np.random.choice(len(iterators), p=probs)
            yield next(iterators[index])
    return tf.data.Dataset.from_generator(lambda: generator(), output_signature=signature)


def sample_edges(edges, banned_edges, neighbours_per_sample):
    if len(edges) > neighbours_per_sample:
        chosen_indexes = np.random.choice(len(edges), size=neighbours_per_sample + 1, replace=False)
        chosen_edges = [edges[index] for index in chosen_indexes if edges[index] not in banned_edges]
        return list(itertools.chain(*chosen_edges[:neighbours_per_sample])), 0
    chosen_edges = [edge for edge in edges if edge not in banned_edges]
    missing_edges = [
        (MISSING_ENTITY_TOKEN_ID, MISSING_RELATION_TOKEN_ID)
        for _ in range(neighbours_per_sample - len(chosen_edges))
    ]
    edges = list(itertools.chain(*chosen_edges)) + list(itertools.chain(*missing_edges))
    return edges, len(missing_edges)


def get_existing_graph_edges(data_directory):
    entity_ids = extract_entity_ids(data_directory)
    relation_ids = extract_relation_ids(data_directory)
    edges_func = functools.partial(
        extract_edges_from_file, entity_ids=entity_ids, relation_ids=relation_ids, data_directory=data_directory
    )
    return (
        edges_func(dataset_type=DatasetType.TRAINING) +
        edges_func(dataset_type=DatasetType.VALIDATION) +
        edges_func(dataset_type=DatasetType.TEST)
    )


def iterator_of_samples_to_dataset(iterator, max_samples=None):
    samples = list(iterator) if max_samples is None else itertools.islice(iterator, max_samples)
    return tf.data.Dataset.from_tensor_slices(dict(map_list_of_dicts_to_dict(samples)))


def combine_datasets(list_of_datasets):
    combined_dataset = list_of_datasets[0]
    for dataset in list_of_datasets[1:]:
        combined_dataset = combined_dataset.concatenate(dataset)
    return combined_dataset
