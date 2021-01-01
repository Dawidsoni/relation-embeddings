import subprocess
import argparse
import os


SCRIPTS_ROOT_FOLDER = "../scripts"


def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scripts_filenames', type=str, required=True, nargs='+')
    return parser.parse_args()


def run_scripts(filenames):
    scripts_paths = [os.path.join(SCRIPTS_ROOT_FOLDER, filename) for filename in filenames]
    for script_path in scripts_paths:
        if not os.path.exists(script_path):
            raise ValueError(f"Unable to find a script in location: '{script_path}'")
    for script_path in scripts_paths:
        subprocess.call(script_path)


if __name__ == '__main__':
    training_args = parse_training_args()
    run_scripts(training_args.scripts_filenames)
