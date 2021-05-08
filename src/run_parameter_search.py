import json
import argparse
import random
import os


def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_configs', type=str, required=True, nargs='+')
    parser.add_argument('--gin_bindings', type=str, default=[], nargs='+')
    parser.add_argument('--search_config', type=str, required=True)
    return parser.parse_args()


def get_cartesian_product_of_configs(parameter_configs):
    produced_configs = [{}]
    for parameter_config in parameter_configs:
        configs_with_parameter = []
        parameter_name = parameter_config["parameter_name"]
        for parameter_value in parameter_config["parameter_values"]:
            for produced_config in produced_configs:
                copied_config = produced_config.copy()
                copied_config[parameter_name] = parameter_value
                configs_with_parameter.append(copied_config)
        produced_configs = configs_with_parameter
    return produced_configs


def get_banned_configs(config):
    if "banned_parameter_configs" not in config:
        return []
    produced_configs = []
    for banned_configs in config["banned_parameter_configs"]:
        if "cartesian_product" in banned_configs:
            produced_configs.extend(get_cartesian_product_of_configs(
                banned_configs["cartesian_product"]
            ))
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
    configs = get_cartesian_product_of_configs(config["parameter_configs"])
    banned_configs = get_banned_configs(config)
    return exclude_banned_configs(configs, banned_configs)


def run_experiment(gin_configs, gin_bindings):
    text_of_gin_configs = "--gin_configs " + " ".join([f"\"{config}\"" for config in gin_configs])
    text_of_gin_bindings = "--gin_bindings " + " ".join([f"\"{binding}\"" for binding in gin_bindings])
    os.system(f'python3 ../src/train_model.py {text_of_gin_configs} {text_of_gin_bindings}')


def run_parameter_search(gin_configs, gin_bindings, search_config):
    with open(search_config, mode="r") as file_stream:
        parameter_configs = parse_parameter_configs(json.load(file_stream))
    random.shuffle(parameter_configs)
    for parameter_config in parameter_configs[:10]:
        experiment_bindings = [
            f"{key} = '{value}'" if isinstance(value, str) and value[0] != "@" else f"{key} = {value}"
            for key, value in parameter_config.items()
        ]
        run_experiment(
            gin_configs,
            gin_bindings=(gin_bindings + experiment_bindings),
        )


if __name__ == '__main__':
    training_args = parse_training_args()
    run_parameter_search(training_args.gin_configs, training_args.gin_bindings, training_args.search_config)
