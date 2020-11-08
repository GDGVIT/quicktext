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

def merge_dictb_to_dicta(dicta, dictb):
    """
    In this function dictb overrides dicta
    replacing values in dicta or adding new values to it
    Args:
        dicta (dict): Dictionary with values
        dictb (dict): Dictionary whose values will be written into dicta
    Returns:
        dict: It will return the updated dicta
    """

    for key in dictb:
        dicta[key] = dictb[key]
    
    return dicta