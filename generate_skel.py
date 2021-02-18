import os,sys
import h5py
import numpy as np
import kimimaro
from io_util import readVol
import pickle

def skeletonize(labels, scale=4, const=500, obj_ids=None, dust_size = 100, res = (32,32,30), num_thread = 1):
    if obj_ids is None:
        obj_ids = np.unique(labels)
        obj_ids = list(obj_ids[obj_ids>0])
    skels = kimimaro.skeletonize(
      labels, 
      teasar_params={
        'scale': scale,
        'const': const, # physical units
        'pdrf_exponent': 4,
        'pdrf_scale': 100000,
        'soma_detection_threshold': 1100, # physical units
        'soma_acceptance_threshold': 3500, # physical units
        'soma_invalidation_scale': 1.0,
        'soma_invalidation_const': 300, # physical units
        'max_paths': 50, # default  None
      },
      object_ids= obj_ids, # process only the specified labels
      # object_ids=[ ... ], # process only the specified labels
      # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
      # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
      dust_threshold = dust_size, # skip connected components with fewer than this many voxels
#       anisotropy=(30,30,30), # default True
      anisotropy = res, # default True
      fix_branching=True, # default True
      fix_borders=True, # default True
      progress=True, # default False, show progress bar
      parallel = num_thread, # <= 0 all cpu, 1 single process, 2+ multiprocess
      parallel_chunk_size = 100, # how many skeletons to process before updating progress bar
    )
    return skels

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print('need four arguments: seg_path resolution skel_id output_path')
        print('example: python eval_erl.py yy.h5 30x6x6 -1 xx.pkl ')
        # resolution: xyz-order
        raise ValueError()
    seg_path, resolution, skel_id, output_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    print('load segmentation')
    seg = readVol(seg_path)
    resolution = [int(x) for x in resolution.split('x')]
    skel_id = None if skel_id == '-1' else [int(x) for x in skel_id.split(',')]

    print('skeletonization')
    skels = skeletonize(seg, obj_ids = skel_id, dust_size = 100, res = resolution)
    
    print('save output')
    nodes = [skels[x].vertices for x in skels]
    edges = [skels[x].edges for x in skels]
    if '/' in output_path:
        fn = output_path[:output_path.rfind('/')]
        if not os.path.exists(fn):
            os.makedirs(fn)

    pickle.dump([nodes, edges], open(output_path, 'wb'))
    print('done')


