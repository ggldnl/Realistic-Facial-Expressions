from kaggle.api.kaggle_api_extended import KaggleApi
from urllib.parse import urlsplit
from pathlib import Path
import requests
import zipfile
import gdown
import csv
import os
import re


def read_tsv(file_path, skip_header=False, encoding='utf-8'):
    """
    Reads a TSV (Tab-Separated Values) file and returns a 2D array.

    :param file_path: The path to the TSV file.
    :param skip_header: Skip the first row of the tsv.
    :param encoding: Encoding.

    :returns: A 2D array with as many rows as the lines in the file
                    and as many columns as the elements in each row.
    """

    data = []
    with open(file_path, 'r', newline='', encoding=encoding) as file:
        tsv_reader = csv.reader(file, delimiter='\t')

        if skip_header:
            next(tsv_reader)

        for row in tsv_reader:
            data.append(row)

    return data


def extract_zip(zip_path, directory):
    """
    Extracts a zip file to the specified directory.

    If the specified directory does not exist, it will be created. The function
    attempts to handle exceptions related to file access, missing files, and
    corrupted zip files by printing an error message. For more detailed error
    handling, consider raising exceptions or implementing additional logic based
    on the type of errors.

    :param zip_path: The path of the zip file to be extracted.
    :param directory: The destination path where the zip file's contents will be extracted.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(directory)

        print(f"Extraction of {zip_path} to {directory} completed successfully.")
    except FileNotFoundError:
        print(f"The file {zip_path} does not exist.")
    except zipfile.BadZipFile:
        print(f"The file {zip_path} is not a zip file or is corrupted.")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_resource(url, filepath):
    """
    Download a resource from a given URL and save it to the specified file path.

    :param url: The URL of the resource to download.
    :param filepath: The file path where the downloaded resource will be saved.

    :return: True if the download was successful, False otherwise.
    """

    parent_folder = os.path.dirname(filepath)
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        with open(filepath, "wb") as file:
            file.write(response.content)
        return True
    else:
        return False


def download_google_drive(url, filepath):
    """
    Download a file from Google Drive given its URL or file ID and save it to the specified file path.

    :param url: The URL or file ID of the file to download from Google Drive.
    :param filepath: The file path where the downloaded file will be saved.

    :returns: True if the download was successful, False otherwise.
    """

    """
    file_id = url_or_file_id
    if is_url(url_or_file_id):
        file_id = extract_google_drive_file_id(url_or_file_id)
    """

    try:
        # Download the file from Google Drive
        gdown.download(url, str(filepath), fuzzy=True, quiet=False)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def download_kaggle(dataset_name, download_dir):
    """
    Download a dataset from Kaggle given its name and save & extract it to the specified file path.

    :param dataset_name: The dataset name of the dataset to download from Google Drive.
    :param download_dir: The file path where the downloaded file will be saved and extracted.

    :return: True (TODO: fix this)
    """

    # Get the value of the HOME environment variable
    home_directory = os.getenv('HOME')

    kaggle_api_path = os.path.join(home_directory, '.kaggle/kaggle.json')

    if not os.path.exists(kaggle_api_path):
        raise RuntimeError("Unable to find Kaggle API key (missing '~/.kaggle/kaggle.json').")

    # Initialize Kaggle API
    api = KaggleApi()
    # Authenticate with your Kaggle credentials
    api.authenticate()

    # Create the directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Download the dataset
    api.dataset_download_files(dataset_name, path=download_dir, unzip=True)

    return True


def is_local_path(path):
    """
    Check if a given path is a local file path.

    :param path: The path to be checked.

    :returns: True if the path is a local file path, False otherwise.
    """
    scheme = urlsplit(path).scheme
    return scheme == '' or scheme.lower() in {'file', 'localhost'}


def is_url(string):
    """
    Check if a given string is a URL.

    :param string: The string to be checked.

    :returns: True if the string is a URL, False if it's not.
    """
    scheme = urlsplit(string).scheme
    return scheme.lower() not in {'', 'file'}


def extract_google_drive_file_id(url):
    """
    Given a Google Drive URL, returns the file id.

    :param url: Google Drive URL.

    :returns: File ID.
    """
    pattern = r"/file/d/([-\w]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None


def ensure_path_object(path):
    """
    Ensures the given object is a pathlib.Path instance.

    :param path: Object to check.

    :returns: A pathlib.Path object conversion of the input.
    """
    if not isinstance(path, Path):
        path = Path(path)
    return path


def remove(path):
    """
    Just a wrapper for the underlying os function.

    :param path: Path of the object to be removed.
    """
    os.remove(path)