from quicktext.imports import *

def read_yaml(path):
    """
    This function reads a YAML file
    Args:
        path (str): Path to YAML file
    Returns:
        dict: Parsed YAML file
    """

    yaml_file = open(path) 
    parsed_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
    edict = EasyDict(parsed_file)

    return edict