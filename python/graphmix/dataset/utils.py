import numpy as np
import pickle
import os
import ssl
import urllib.request
import zipfile

def download_url(url, folder):
    """Downloads the content of an URL to a specific folder.
    Args:
        url (string): The url.
        folder (string): The folder.
    """
    filename = url.rpartition('/')[2].split('?')[0]
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        return path
    print('Downloading', url)
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(folder)))
    except Exception:
        pass
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        f.write(data.read())
    print("Downloaded", path)
    return path

def extract_zip(path, folder):
    """Extracts a zip archive to a specific folder.
    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
    """
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)

def process_graph(graph_dict):
    row, col = [], []
    for key, value in graph_dict.items():
        value = list(set(value))
        row += [key] * len(value)
        col += sorted(value)
    return np.vstack([row, col])
