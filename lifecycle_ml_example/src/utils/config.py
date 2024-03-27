import yaml
from pathlib import Path


CONFIG_FOLDER_NAME = "conf"


def load_config():
    """Load config file"""
    root_path = Path(__file__).parent.parent.parent

    with open(root_path / f"{CONFIG_FOLDER_NAME}/config.yaml", mode="r") as fileyaml:
        config = yaml.load(fileyaml, Loader=yaml.FullLoader)

    # load secrets
    secrets = load_secrets()

    # add secrets (dict) to config
    config.update(secrets)

    # set env variables
    set_env_var(config)

    return config


def load_secrets():
    """Load secrets"""
    root_path = Path(__file__).parent.parent.parent
    # check if secrets.yaml exists
    if (root_path / f"{CONFIG_FOLDER_NAME}/secrets.yaml").exists():

        with open(root_path / f"{CONFIG_FOLDER_NAME}/secrets.yaml", mode="r") as fileyaml:
            secrets = yaml.load(fileyaml, Loader=yaml.FullLoader)
        return secrets

    return {}


def set_env_var(config):
    """Set env variables"""
    pass


if __name__ == "__main__":
    config = load_config()
    print(config)
