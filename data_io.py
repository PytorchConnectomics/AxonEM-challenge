import sys
import pickle
import h5py
import numpy as np


def read_vol(filename, dataset_name=None, chunk_id=0, chunk_num=1):
    """
    The function `read_vol` reads a volume from a file, either in HDF5 or TIFF format.

    :param filename: The name of the file to be read. It can be either a .h5 file or a .tif/.tiff file
    :param dataset_name: The `dataset_name` parameter is used to specify the name of the dataset within
    the HDF5 file that you want to read. If the HDF5 file contains multiple datasets, you can use this
    parameter to specify which dataset you want to read. If `dataset_name` is not provided, the function
    :param chunk_id: The `chunk_id` parameter is used to specify the index of the chunk to read from a
    file. Chunks are smaller portions of a dataset that can be read separately. By specifying the
    `chunk_id`, you can read a specific chunk of data from the file, defaults to 0 (optional)
    :param chunk_num: The parameter `chunk_num` is used to specify the total number of chunks into which
    the data is divided. It is used in conjunction with the `chunk_id` parameter to read a specific
    chunk of data from a file. By default, `chunk_num` is set to 1, indicating that, defaults to 1
    (optional)
    :return: the result of either the `read_h5` function or the `volread` function, depending on the
    file type of the input filename.
    """
    if ".h5" in filename:
        return read_h5(
            filename, dataset_name, chunk_id=chunk_id, chunk_num=chunk_num
        )
    elif ".tif" in filename or ".tiff" in filename:
        from imageio import volread

        return volread(filename)
    else:
        raise ValueError("cannot recognize input file type:", filename)


def read_pkl(filename):
    """
    The function `read_pkl` reads a pickle file and returns a list of the objects stored in the file.

    :param filename: The filename parameter is a string that represents the name of the file you want to
    read. It should include the file extension, such as ".pkl" for a pickle file
    :return: a list of objects that were read from the pickle file.
    """
    data = []
    with open(filename, "rb") as fid:
        while True:
            try:
                data.append(pickle.load(fid))
            except EOFError:
                break
    return data


def write_pkl(filename, content):
    """
    The function `write_pkl` writes content to a pickle file with the specified filename and protocol.

    :param filename: The filename parameter is a string that specifies the name of the file to write the
    content to. It should include the file extension, such as ".pkl" for a pickle file
    :param content: The `content` parameter is the data that you want to write to the pickle file. It
    can be of any type that is supported by the pickle module, such as lists, dictionaries, strings,
    numbers, etc
    """
    with open(filename, "wb") as fid:
        if isinstance(content, (list,)):
            for val in content:
                pickle.dump(val, fid)
        else:
            pickle.dump(content, fid)


def read_h5(filename, dataset_names=None, chunk_id=0, chunk_num=1):
    """
    The function `read_h5` reads data from an HDF5 file, either the entire dataset or a specified chunk,
    and returns the data as a numpy array or a list of numpy arrays.

    :param filename: The filename parameter is the name of the HDF5 file that you want to read
    :param dataset_names: The dataset_name parameter is used to specify the name of the dataset(s) you
    want to read from the HDF5 file. If dataset_name is not provided, the function will read all
    datasets in the file
    :param chunk_id: The `chunk_id` parameter is used to specify the index of the chunk of data to read
    from the dataset. It is used in conjunction with the `chunk_num` parameter to divide the dataset
    into chunks and read a specific chunk of data. The default value of `chunk_id` is 0, defaults to 0
    (optional)
    :param chunk_num: The parameter "chunk_num" is used to specify the number of chunks to divide the
    dataset into. This is useful when dealing with large datasets that cannot fit into memory all at
    once. By dividing the dataset into chunks, you can read and process smaller portions of the data at
    a time, defaults to 1 (optional)
    :return: the dataset(s) from the specified HDF5 file. If only one dataset is specified, it returns
    that dataset as a numpy array. If multiple datasets are specified, it returns a list of numpy
    arrays, each corresponding to a dataset.
    """
    fid = h5py.File(filename, "r")
    if dataset_names is None:
        dataset_names = fid.keys() if sys.version[0] == "2" else list(fid)

    out = [None] * len(dataset_names)
    for dataset_id, dataset_name in enumerate(dataset_names):
        if chunk_num == 1:
            out[dataset_id] = np.array(fid[dataset_name])
        else:
            num_z = int(np.ceil(fid[dataset_name].shape[0] / float(chunk_num)))
            out[dataset_id] = np.array(
                fid[dataset_name][chunk_id * num_z : (chunk_id + 1) * num_z]
            )
    fid.close()
    return out[0] if len(out) == 1 else out


def get_volume_size_h5(filename, dataset_name=None):
    """
    The function `get_volume_size_h5` returns the size of a dataset in an HDF5 file, or the size of the
    first dataset if no dataset name is provided.

    :param filename: The filename parameter is the name of the HDF5 file that you want to read
    :param dataset_name: The parameter `dataset_name` is an optional argument that specifies the name of
    the dataset within the HDF5 file. If it is not provided, the function will retrieve the first
    dataset in the file and return its shape as the volume size
    :return: the size of the volume as a list.
    """
    volume_size = []
    fid = h5py.File(filename, "r")
    if dataset_name is None:
        dataset_name = fid.keys() if sys.version[0] == "2" else list(fid)
        if len(dataset_name) > 0:
            volume_size = fid[dataset_name[0]].shape
    fid.close()
    return volume_size
