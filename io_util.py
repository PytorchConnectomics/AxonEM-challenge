import sys
import numpy as np

def readVol(filename, datasetname=None):
    if '.h5' in filename:
        return readH5(filename, datasetname)
    elif '.tif' in filename or '.tiff' in filename:
        from imageio import volread
        return volread(filename)
    else:
        raise ValueError('cannot recognize input file type:', filename)

def readH5(filename, datasetname=None):
    import h5py
    fid = h5py.File(filename,'r')
    if datasetname is None:
        if sys.version[0]=='2': # py2
            datasetname = fid.keys()
        else: # py3
            datasetname = list(fid)
    if len(datasetname) == 1:
        datasetname = datasetname[0]
    if isinstance(datasetname, (list,)):
        out=[None]*len(datasetname)
        for di,d in enumerate(datasetname):
            out[di] = np.array(fid[d])
        return out
    else:
        return np.array(fid[datasetname])


