import sys
import numpy as np
import h5py
from imageio import volread
        

def read_vol(filename, datasetname=None, chunk_id=0, chunk_num=1):
    if '.h5' in filename:
        return read_h5(filename, datasetname, chunk_id=0, chunk_num=1)
    elif '.tif' in filename or '.tiff' in filename:
        return volread(filename)
    else:
        raise ValueError('cannot recognize input file type:', filename)


def read_h5(filename, datasetname=None, chunk_id=0, chunk_num=1):
    fid = h5py.File(filename, 'r')
    if datasetname is None:        
        datasetname = fid.keys() if sys.version[0]=='2' else list(fid)    

    out = [None] * len(datasetname)
    for di, d in enumerate(datasetname):            
        if chunk_num == 1:
            out[di] = np.array(fid[d])
        else:
            num_z = int(np.ceil(fid[d].shape[0]/float(chunk_num)))
            out[di] = np.array(fid[d][chunk_id*num_z: (chunk_id + 1) * num_z])

    return out[0] if len(out) == 1 else out