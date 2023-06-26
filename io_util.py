import sys
import numpy as np
import h5py
import pickle
from imageio import volread
        

def read_vol(filename, datasetname=None, chunk_id=0, chunk_num=1):
    if '.h5' in filename:
        return read_h5(filename, datasetname, chunk_id=0, chunk_num=1)
    elif '.tif' in filename or '.tiff' in filename:
        return volread(filename)
    else:
        raise ValueError('cannot recognize input file type:', filename)

def read_pkl(filename):
      data = []
      with open(filename, "rb") as f:
          while True:
              try:
                  data.append(pickle.load(f))
              except:
                  break
      return data

def writepkl(filename, content, protocol=pickle.HIGHEST_PROTOCOL):
    with open(filename, "wb") as f:
        if isinstance(content, (list,)):
            for val in content:
                pickle.dump(val, f)
        else:
            pickle.dump(content, f)


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
    fid.close()
    return out[0] if len(out) == 1 else out


def get_volume_size_h5(filename, datasetname=None):
    volume_size = [] 
    fid = h5py.File(filename, 'r')
    if datasetname is None:        
        datasetname = fid.keys() if sys.version[0]=='2' else list(fid)    
        if len(datasetname) > 0:
            volume_size = fid[datasetname[0]].shape
    fid.close()

    return volume_size
