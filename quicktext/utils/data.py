from quicktext.imports import *


def load_from_directory(directory):
    """
    This function loads text data from directory
    Dir names are inferred as class labels
    All files inside the directories are read 

    Args:
    directory (str): Directory containing the data
    Return:
    dictionary: With keys target, data
            data is a list of text files
            target is corresponding label 
    """

    labels = os.listdir(directory)

    data = []
    target = []
    for idx, label in enumerate(labels):
        cur_dir = os.path.join(directory, label)
        for file in os.listdir(cur_dir):
            text_file = open(os.path.join(cur_dir, file), "rb")
            text = text_file.read()
            text_file.close()

            data.append(text)
            target.append(idx)

    target_names = labels

    payload = EasyDict({"data": data, "target": target, "target_names": target_names})
    return payload
