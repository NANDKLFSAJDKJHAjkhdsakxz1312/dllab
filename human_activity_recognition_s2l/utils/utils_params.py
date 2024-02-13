import os
import datetime

def gen_run_folder(path_model_id=""):
    """Generates a folder structure for a model run with unique identifiers and paths for logging and checkpoints.

    Args:
        path_model_id (str): Optional. An identifier for the model run. If not provided, a timestamp is used.

    Returns:
        dict: A dictionary containing paths for the model ID, training logs, evaluation logs, training checkpoints, evaluation checkpoints, gin configuration, training summary, and profiler summary.
    """
    run_paths = dict()
    path_model_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "experiments")
    )

    # Function to find if the directory exists; if not, creates a new one
    def find_file(file_dir, path_model_id):
        target_dir_path = os.path.join(file_dir, path_model_id)
        if os.path.isdir(target_dir_path):
            return True, target_dir_path
        else:
            return False, path_model_id

    file_exist, path_model_id = find_file(path_model_root, path_model_id)

    if not file_exist:
        date_creation = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
        run_id = "run_" + date_creation
        if path_model_id:
            run_id += "_" + path_model_id
        path_model_id = run_id  # Fixed logic error in original code
        run_paths["path_model_id"] = os.path.join(path_model_root, run_id)
    else:
        run_paths["path_model_id"] = path_model_id

    # Define paths for logs, checkpoints, gin configuration, and summaries
    paths_keys = ["path_logs_train", "path_logs_eval", "path_ckpts_train", "path_ckpts_eval", "path_gin",
                  "path_summary_train", "path_summary_profiler"]
    paths_values = ["logs/run.log", "logs/eval/run.log", 'ckpts', "ckpts/eval", "config_operative.gin",
                    'summary/train', 'summary']
    run_paths.update({k: os.path.join(run_paths["path_model_id"], v) for k, v in zip(paths_keys, paths_values)})

    # Create necessary directories
    for key, path in run_paths.items():
        if "path" in key and "logs" not in key:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    # Ensure log files exist
    for log_key in [k for k in run_paths if "logs" in k]:
        os.makedirs(os.path.dirname(run_paths[log_key]), exist_ok=True)
        open(run_paths[log_key], "a").close()  # Touch file

    return run_paths

def save_config(path_gin, config):
    """Saves the gin configuration to a file.

    Args:
        path_gin (str): Path to the gin configuration file.
        config (str): The gin configuration to be saved.
    """
    with open(path_gin, 'w') as f_config:
        f_config.write(config)

def gin_config_to_readable_dictionary(gin_config: dict):
    """Converts gin configuration to a more readable dictionary format, useful for logging.

    Args:
        gin_config (dict): The gin configuration dictionary.

    Returns:
        dict: A cleaned and parsed dictionary of the gin configuration.
    """
    data = {}
    for key, values in gin_config.items():
        name = key[1].split(".")[1]  # Extract name from the tuple key
        for k, v in values.items():
            data[f"{name}/{k}"] = v

    return data
