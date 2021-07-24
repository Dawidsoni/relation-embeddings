import functools
import os
import numpy as np

from optimization.datasets import MaskedEntityOfEdgeDataset, DatasetType
from optimization.similar_entities_producer import SimilarEntitiesProducer, SimilarEntitiesProducerConfig, \
    EntitiesSimilarityType


def compute_and_save_dataset_similarities(dataset, path_to_save_data, similarity_type):
    similar_entities_producer = SimilarEntitiesProducer(SimilarEntitiesProducerConfig(
        cache_path=path_to_save_data,
        dataset_name=os.path.basename(dataset.data_directory),
        graph_edges=dataset.graph_edges,
        entities_count=dataset.entities_count,
        relations_count=dataset.relations_count,
        similarity_type=similarity_type,
        similar_entities_count=50,
    ))
    similar_entities_producer.produce_similar_entities(
        entity_ids=np.arange(dataset.entities_count, dtype=np.int32)
    )
    print(f"Finished computing similarities for path {path_to_save_data} and similarity type: {similarity_type}")


def run_script():
    compute_fb15_similarities = functools.partial(
        compute_and_save_dataset_similarities,
        dataset=MaskedEntityOfEdgeDataset(dataset_type=DatasetType.TRAINING, data_directory="../data/FB15k-237"),
        path_to_save_data="../similarities/FB15k-237",
    )
    compute_fb15_similarities(similarity_type=EntitiesSimilarityType.INPUT_RELATIONS)
    compute_fb15_similarities(similarity_type=EntitiesSimilarityType.OUTPUT_RELATIONS)
    compute_wn18rr_similarities = functools.partial(
        compute_and_save_dataset_similarities,
        dataset=MaskedEntityOfEdgeDataset(dataset_type=DatasetType.TRAINING, data_directory="../data/WN18RR"),
        path_to_save_data="../similarities/WN18RR",
    )
    compute_wn18rr_similarities(similarity_type=EntitiesSimilarityType.INPUT_RELATIONS)
    compute_wn18rr_similarities(similarity_type=EntitiesSimilarityType.OUTPUT_RELATIONS)


run_script()
