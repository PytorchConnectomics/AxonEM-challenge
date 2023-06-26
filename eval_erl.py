import sys
from funlib import evaluate
import numpy as np
import networkx as nx
from io_util import read_vol
import pickle


def skeleton_to_networkx(nodes, edges, resolution, seg_list=None, do_node_out=False):
    # convert skeleton into networkx object
    # if a list of segments exist, compute node-segment assignment
    # node: physical unit
    if seg_list is None:
        seg_list = []
    gt_graph = nx.Graph()
    node_segment_lut = [{}]*len(seg_list)
    cc = 0
    node_out = [None] * len(nodes)
    for k in range(len(nodes)):
        if len(edges[k]) == 0:
            continue
        node_arr = nodes[k].astype(int)
        # augment the node index
        edge_arr = edges[k] + cc
        for node in node_arr:
            gt_graph.add_node(cc, skeleton_id=k,
                              z=node[0]*resolution[0],
                              y=node[1]*resolution[1],
                              x=node[2]*resolution[2])
            for i in range(len(seg_list)):
                node_segment_lut[i][cc] = seg_list[i][node[0],
                                                      node[1],
                                                      node[2]]
            cc += 1
        for edge in edge_arr:
            gt_graph.add_edge(edge[0], edge[1])
        if do_node_out:
            node_out[k] = np.hstack([np.arange(cc - node_arr.shape[0], cc).reshape(-1, 1),
                                     node_arr])
    if do_node_out:
        node_out = np.vstack(node_out)
        return gt_graph, node_segment_lut, node_out
    return gt_graph, node_segment_lut


def compute_node_segment_lut(node_out, seg_list, chunk_num=1, dtype=np.uint32):
    node_segment_lut = [np.zeros([node_out[:, 0].max() + 1, 4], dtype)] * len(seg_list)
    for seg_id in range(len(seg_list)):
        node_segment_lut[seg_id][node_out[:, 0]] = seg_list[seg_id][node_out[:, 1],
                                                                    node_out[:, 2],
                                                                    node_out[:, 3]]
    return node_segment_lut


def compute_node_segment_lut_byname(node_out, seg_name_list, chunk_num=1, dtype=np.uint32):
    node_segment_lut = [np.zeros([node_out[:, 0].max() + 1, 4], dtype)] * len(seg_name_list)
    for seg_id, seg_name in enumerate(seg_name_list):
        assert '.h5' in seg_name
        start_z = 0
        for chunk_id in range(chunk_num):
            seg = read_vol(seg_name, None, chunk_id, chunk_num)
            last_z = start_z + seg.shape[0]
            pts = node_out[(node_out[:, 1] >= start_z) *
                           (node_out[:, 1] < last_z)]
            node_segment_lut[seg_id][pts[:, 0]] = seg[pts[:, 1] - start_z,
                                                      pts[:, 2],
                                                      pts[:, 3]]
            start_z = last_z
    return node_segment_lut


def compute_erl(gt_nodes, gt_edges, pred_seg, resolution=None,
                merge_threshold=0, node_segment_lut=None):
    if resolution is None:
        resolution = [30, 6, 6]
    if node_segment_lut is not None:
        gt_graph, _ = skeleton_to_networkx(gt_nodes, gt_edges, resolution)
    else:
        gt_graph, node_segment_lut = skeleton_to_networkx(gt_nodes, gt_edges, resolution, [pred_seg])
    scores = evaluate.expected_run_length(
                    skeletons=gt_graph,
                    skeleton_id_attribute='skeleton_id',
                    edge_length_attribute='length',
                    node_segment_lut=node_segment_lut[0],
                    merge_threshold=merge_threshold,
                    skeleton_position_attributes=['z', 'y', 'x'])
    print('ERL:', scores)


if __name__ == "__main__":
    assert len(sys.argv) >= 3, print('need at least three arguments: skel_path, seg_path resolution merge_thresholdhold\n example: python eval_erl.py xx.pkl yy.h5 30x6x6')
    # resolution and node should match
    skeleton_path, seg_path, resolution = sys.argv[1], sys.argv[2], sys.argv[3]
    # threshold for number of merges: to be robust to skeleton points aliasing
    merge_threshold = 0 if len(sys.argv) < 4 else int(sys.argv[4])
    node_unit = 'voxel' if len(sys.argv) < 5 else sys.argv[5]

    resolution = [int(x) for x in resolution.split('x')]
    # node unit: voxel
    skeleton = pickle.load(open(skeleton_path, 'rb'), encoding="latin1")
    gt_nodes, gt_edges = skeleton.vertices, skeleton.edges
    if node_unit == 'physical':
        for i in range(len(gt_nodes)):
            if len(gt_nodes[i]) > 0:
                gt_nodes[i] = gt_nodes[i] / resolution

    pred_seg = read_vol(seg_path)

    compute_erl(gt_nodes, gt_edges, pred_seg, resolution, merge_threshold)
