import sys
from funlib import evaluate
import numpy as np
import networkx as nx
from io_util import read_vol
import pickle


def skeleton_to_networkx(nodes, edges, resolution, do_node_position=False):
    # convert skeleton into networkx object
    # if a list of segments exist, compute node-segment assignment
    # node: physical unit
    gt_graph = nx.Graph()
    cc = 0
    node_position = [None] * len(nodes)
    for k in range(len(nodes)):
        if len(edges[k]) == 0:
            continue
        node_arr = nodes[k].astype(np.uint16)
        # augment the node index
        edge_arr = edges[k] + cc
        for node in node_arr:
            # unit: physical
            gt_graph.add_node(cc, skeleton_id=k,
                              z=node[0],
                              y=node[1],
                              x=node[2])
            cc += 1
        for edge in edge_arr:
            gt_graph.add_edge(edge[0], edge[1])
        if do_node_position:
            node_position[k] = node_arr // resolution
    if do_node_position:
        node_position = np.vstack(node_position)
        return gt_graph, node_position
    return gt_graph


def compute_node_segment_lut(node_position, seg_list, dtype=np.uint32):
    node_segment_lut = [np.zeros([node_position.shape[0], 4], dtype)] * len(seg_list)
    for seg_id in range(len(seg_list)):
        node_segment_lut[seg_id] = seg_list[seg_id][node_position[:, 0],
                                                    node_position[:, 1],
                                                    node_position[:, 2]]
    return node_segment_lut


def compute_node_segment_lut_lowmem(node_position, seg_name_list, chunk_num=1, dtype=np.uint32):
    # low memory version of the lut computation
    # node coord: zyx
    node_segment_lut = [np.zeros(node_position.shape[0], dtype)] * len(seg_name_list)
    for seg_id, seg_name in enumerate(seg_name_list):
        assert '.h5' in seg_name
        start_z = 0
        for chunk_id in range(chunk_num):
            seg = read_vol(seg_name, None, chunk_id, chunk_num)
            last_z = start_z + seg.shape[0]
            ind = (node_position[:, 0] >= start_z) * (node_position[:, 0] < last_z)
            pts = node_position[ind]
            node_segment_lut[seg_id][ind] = seg[pts[:, 0] - start_z,
                                                            pts[:, 1],
                                                            pts[:, 2]]
            start_z = last_z
    return node_segment_lut


def compute_erl(gt_graph, node_segment_lut, merge_threshold=0):
    scores = [None] * len(node_segment_lut)
    for lid, lut in enumerate(node_segment_lut):
        scores[lid] = evaluate.expected_run_length(
                    skeletons=gt_graph,
                    skeleton_id_attribute='skeleton_id',
                    edge_length_attribute='length',
                    node_segment_lut=lut,
                    merge_thres=merge_threshold,
                    skeleton_position_attributes=['z', 'y', 'x'])
    return scores
