

import os
from urllib.request import urlopen
import gzip

def maybe_download(filename, directory, url):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        print("Download {} ...".format(filepath))
        data = urlopen(url).read()
        with open(filepath, 'wb') as f:
            f.write(data)
    return filepath

def gunzip(in_path):
    out_path = in_path.rsplit('.', 1)[0]
    with gzip.open(in_path, 'rb') as in_file:
        content = in_file.read()
        with open(out_path, 'wb') as out_file:
            out_file.write(content)
