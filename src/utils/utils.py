import yaml
from pprint import PrettyPrinter


def load_config(path):
    """
    Load config file from path

    Note
    ----
    Only YAML files are supported.

    Parameters
    ----------
    path : str
        Path to config file
    """
    with open(path) as file:
        config = yaml.safe_load(file)

    return config


def print_config(config):
    """
    Prints a YAML config in a pretty way.

    Parameters
    ----------
    config : dict
        The config dictionary to be printed.
    """
    pp = PrettyPrinter()
    pp.pprint(config)


def replace_slashes(input_string, replace_with="_"):
    """
    Replace the slashes (/ and \) in a string.
    This can be useful when saving matplotlib plots
    to avoid path error.
    """
    return input_string.replace("/", replace_with).replace("\\", replace_with)
