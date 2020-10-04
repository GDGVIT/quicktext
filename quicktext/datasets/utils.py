from quicktext.imports import *


def get_dataset(name):
    """
    This function gets dataset from drive
    Args:
        name (str): Name of dataset
    Returns:
        TextClassifierData: An object of TextClassifierData, this class inherits 
                            torch Dataset
    """

    id_map = {"imdb_reviews": "1JOrPeARGs1o7R_0zNdoglV1uywVjVYCb"}

    if name not in id_map:
        raise Exception("Dataset not found")

    dataset_id = id_map[name]
    dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "files")

    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    save_path = os.path.join(dataset_dir, "{}.zip".format(name))

    if not file_exists(dataset_dir, "{}.zip".format(name)):
        url = "https://drive.google.com/uc?export=download&id={}".format(dataset_id)
        gdown.download(url, save_path, quiet=False)

    # Extract the pdf
    extract_zip(save_path, dataset_dir)

    f = open(os.path.join(dataset_dir, "data.pkl"), "rb")
    data = pickle.load(f)
    f.close()

    return data


def extract_zip(path_to_zip_file, directory_to_extract_to):
    """
    This function extracts zip files
    Args:
        path_to_zip_file (str): Path to the zip file to extract
        directory_to_extract_to (str): Directory to extract to
    Returns:
        None
    """

    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def file_exists(dir, file_name):
    _path = os.path.join(dir, file_name)
    return os.path.exists(_path)
