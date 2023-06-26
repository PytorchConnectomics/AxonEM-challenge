import os, sys
import argparse
import numpy as np
from io_util import *
from eval_erl import *


def test_axonEM(gt_graph, node_out, pred_seg_path, num_chunk=1):
    node_segment_lut_chunk = compute_node_segment_lut_byname(node_out, [pred_seg_path], num_chunk)
    scores = compute_erl(gt_graph, node_segment_lut_chunk)
    for sid, score in enumerate(scores):
        print(f'ERL for seg {sid}: {score}')


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluation on AxonEM")
    parser.add_argument("-s", "--seg-path", type=str, help="path to the segmentation prediction", required=True)
    parser.add_argument("-g", "--gt-stats-path", type=str, help="path to ground truth skeleton statistics", default="")
    parser.add_argument("-gp", "--gt-skel-path", type=str, help="path to ground truth seg", default="")
    parser.add_argument("-gr", "--gt-skel-resolution", type=str, help="gt skel resolution", default="60,64,64")
    parser.add_argument("-v", "--volume-name", type=str, help="volume name {human, mouse}", required=True)
    parser.add_argument("-c", "--num-chunk", type=int, help="number of chunks to process the volume", default=1)
    args = parser.parse_args()
    pred_seg_shape = get_volume_size_h5(args.seg_path)
    gt_shape = np.array([1000,2048,2048])
    if args.volume_name == 'mouse':
        gt_shape[0] = 750
    assert np.abs(gt_shape-pred_seg_shape).max()==0, print(f'segmentation volume for {args.volume_name} volume has to be {gt_shape}')

    return args


if __name__ == "__main__":
    # python test_axonEM.py -s db/30um_human/axon_release/gt_16nm.h5 -g db/30um_human/axon_release/gt_16nm_skel_stats.p -v human -c 1 
    args = get_arguments()
    # get stats
    if args.gt_stats_path != "":
        gt_graph, node_out = read_pkl(args.gt_stats_path) 
    else:
        gt_skel = read_pkl(args.gt_skel_path)[0]
        gt_nodes = [gt_skel[x].vertices for x in gt_skel]
        gt_edges = [gt_skel[x].edges for x in gt_skel]

        gt_resolution = [int(x) for x in args.gt_skel_resolution.split(',')]
        gt_graph, _, node_out = skeleton_to_networkx(gt_nodes, gt_edges, gt_resolution, [], True)
        gt_stats_path = args.gt_skel_path[:args.gt_skel_path.rfind('.')] + '_stats.p'
        writepkl(gt_stats_path, [gt_graph, node_out])

    # compute erl
    test_axonEM(gt_graph, node_out, args.seg_path)
