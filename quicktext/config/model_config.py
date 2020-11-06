from quicktext.imports import *


def read_yaml(model_name):

    yaml_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "yaml")
    path = os.path.join(yaml_dir, "{}.yaml".format(model_name))
    file = open(path)
    config_dict = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
    return config_dict


def BiLSTMConfig():
    """
    This function returns config file for BiLSTM
    """
    config_dict = read_yaml("bilstm")
    return EasyDict(config_dict)
