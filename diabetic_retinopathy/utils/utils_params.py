import os
import datetime

def gen_run_folder(path_model_id=""):
    run_paths = dict()
    path_model_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "experiments")
    )
    # find if there is already a file, if not then crate a new one
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
        run_id = path_model_id
        run_paths["path_model_id"] = os.path.join(path_model_root, run_id)
    else:
        run_paths["path_model_id"] = path_model_id

    run_paths["path_logs_train"] = os.path.join(run_paths["path_model_id"], "logs", "run.log")
    run_paths["path_logs_eval"] = os.path.join(run_paths["path_model_id"], "logs", "eval", "run.log")
    run_paths["path_ckpts_train"] = os.path.join(run_paths["path_model_id"], 'ckpts')
    run_paths["path_ckpts_eval"] = os.path.join(run_paths["path_model_id"], "ckpts", "eval")
    run_paths["path_gin"] = os.path.join(run_paths["path_model_id"], "config_operative.gin")
    run_paths['path_summary_train'] = os.path.join(run_paths["path_model_id"], 'summary', 'train')
    run_paths['path_summary_profiler'] = os.path.join(run_paths['path_model_id'], 'summary')

    # Create folders
    for k, v in run_paths.items():
        if any([x in k for x in ["path_model", "path_ckpts", "path_summary"]]):
            if not os.path.exists(v):
                os.makedirs(v, exist_ok=True)

    # Create files
    for k, v in run_paths.items():
        if any([x in k for x in ["path_logs"]]):
            if not os.path.exists(v):
                os.makedirs(os.path.dirname(v), exist_ok=True)
                with open(v, "a"):
                    pass  # atm file creation is sufficient

    return run_paths


def save_config(path_gin, config):
    with open(path_gin, 'w') as f_config:
        f_config.write(config)

def gin_config_to_readable_dictionary(gin_config: dict):
    """
    Parses the gin configuration to a dictionary. Useful for logging to e.g. W&B
    :param gin_config: the gin's config dictionary. Can be obtained by gin.config._OPERATIVE_CONFIG
    :return: the parsed (mainly: cleaned) dictionary
    """
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[1]
        values = gin_config[key]
        for k, v in values.items():
            data["/".join([name, k])] = v

    return data