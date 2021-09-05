import json
import argparse
import random
import os
import numpy as np


def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_configs', type=str, required=True, nargs='+')
    parser.add_argument('--gin_bindings', type=str, default=[], nargs='+')
    parser.add_argument('--search_config', type=str, required=True)
    return parser.parse_args()


def get_list_of_unequal_params(config):
    if "banned_parameter_configs" not in config:
        return []
    list_of_unequal_params = []
    for banned_configs in config["banned_parameter_configs"]:
        if "any_unequal" in banned_configs:
            list_of_unequal_params.append(banned_configs["any_unequal"])
    return list_of_unequal_params


def get_cartesian_product_of_configs(parameter_configs, list_of_unequal_params=(), max_sampled_params=None):
    produced_configs = [{}]
    for parameter_config in parameter_configs:
        configs_with_parameter = []
        parameter_name = parameter_config["parameter_name"]
        parameter_values = parameter_config["parameter_values"]
        if max_sampled_params is not None and len(parameter_values) > max_sampled_params:
            parameter_values = [
                parameter_values[index]
                for index in np.random.choice(len(parameter_values), max_sampled_params, replace=False)
            ]
        for parameter_value in parameter_values:
            for produced_config in produced_configs:
                copied_config = produced_config.copy()
                copied_config[parameter_name] = parameter_value
                skip_config = False
                for unequal_params in list_of_unequal_params:
                    unequal_values = [copied_config[param] for param in unequal_params if param in copied_config]
                    if len(unequal_values) > 0 and any([value != unequal_values[0] for value in unequal_values]):
                        skip_config = True
                if not skip_config:
                    configs_with_parameter.append(copied_config)
        produced_configs = configs_with_parameter
    return produced_configs


def get_banned_configs(config, parsed_configs):
    if "banned_parameter_configs" not in config:
        return []
    produced_configs = []
    for banned_configs in config["banned_parameter_configs"]:
        if "cartesian_product" in banned_configs:
            produced_configs.extend(get_cartesian_product_of_configs(
                banned_configs["cartesian_product"]
            ))
        elif "any_unequal" in banned_configs:
            pass
        else:
            raise ValueError(f"Unexpected node in: {banned_configs}")
    return produced_configs


def exclude_banned_configs(configs, banned_configs):
    filtered_configs = []
    for config in configs:
        is_banned = False
        for banned_config in banned_configs:
            if all([config[key] == value for key, value in banned_config.items()]):
                is_banned = True
                break
        if not is_banned:
            filtered_configs.append(config)
    return filtered_configs


def parse_parameter_configs(config):
    list_of_unequal_params = get_list_of_unequal_params(config)
    parsed_configs = get_cartesian_product_of_configs(
        config["parameter_configs"], list_of_unequal_params, max_sampled_params=2,
    )
    banned_configs = get_banned_configs(config, parsed_configs)
    return exclude_banned_configs(parsed_configs, banned_configs)


def run_experiment(gin_configs, gin_bindings):
    text_of_gin_configs = "--gin_configs " + " ".join([f"\"{config}\"" for config in gin_configs])
    text_of_gin_bindings = "--gin_bindings " + " ".join([f"\"{binding}\"" for binding in gin_bindings])
    os.system(f'python3 ../src/train_model.py {text_of_gin_configs} {text_of_gin_bindings}')


def run_parameter_search(gin_configs, gin_bindings, search_config):
    with open(search_config, mode="r") as file_stream:
        parameter_configs = parse_parameter_configs(json.load(file_stream))
    random.shuffle(parameter_configs)
    for parameter_config in parameter_configs:
        experiment_bindings = [
            f"{key} = '{value}'" if isinstance(value, str) and value[0] not in "@%" else f"{key} = {value}"
            for key, value in parameter_config.items()
            if not key.startswith("_gin.config")
        ]
        search_specific_configs = [value for key, value in parameter_config.items() if key.startswith("_gin.config")]
        run_experiment(
            gin_configs + search_specific_configs,
            gin_bindings=(gin_bindings + experiment_bindings),
        )


if __name__ == '__main__':
    training_args = parse_training_args()
    run_parameter_search(training_args.gin_configs, training_args.gin_bindings, training_args.search_config)
