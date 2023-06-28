import sys
import numpy as np
from io_util import read_vol
from eval_erl import *
import pickle

def test_volume(skeleton_path, seg_path, resolution, merge_threshold, node_unit):
    # node unit: voxel
    skeleton = pickle.load(open(skeleton_path, 'rb'), encoding="latin1")
    gt_nodes, gt_edges = skeleton.vertices, skeleton.edges
    if node_unit == 'physical':
        for i in range(len(gt_nodes)):
            if len(gt_nodes[i]) > 0:
                gt_nodes[i] = gt_nodes[i] / resolution
    pred_seg = read_vol(seg_path)
    gt_graph, node_segment_lut = skeleton_to_networkx(gt_nodes, gt_edges, resolution, [pred_seg])

    scores = compute_erl(gt_graph, node_segment_lut, merge_threshold)
    for sid, score in enumerate(scores):
        print(f'ERL for seg {sid}: {score}')

if __name__ == "__main__":
    assert len(sys.argv) >= 3, print('need at least three arguments: skel_path, seg_path resolution merge_thresholdhold\n example: python eval_erl.py xx.pkl yy.h5 30x6x6')
    # resolution and node should match
    skeleton_path, seg_path, resolution = sys.argv[1], sys.argv[2], sys.argv[3]
    # threshold for number of merges: to be robust to skeleton points aliasing
    merge_threshold = 0 if len(sys.argv) < 4 else int(sys.argv[4])
    node_unit = 'voxel' if len(sys.argv) < 5 else sys.argv[5]

    resolution = [int(x) for x in resolution.split('x')]

    test_volume(skeleton_path, seg_path, resolution, merge_threshold, node_unit)
