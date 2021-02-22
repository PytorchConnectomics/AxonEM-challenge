import sys,os
from funlib import evaluate
import numpy as np
import networkx as nx
from io_util import readVol
import pickle

def skelToNetworkX(nodes, edges, seg_list, res):
    # for ERL evaluation
    # node: physical unit
    gt_graph = nx.Graph()
    node_segment_lut = [{}]*len(seg_list)
    cc = 0
    for k in range(len(nodes)):
        if len(edges[k]) == 0:
            continue
        node = nodes[k].astype(int)
        edge = edges[k] + cc
        for l in range(node.shape[0]):
            gt_graph.add_node(cc, skeleton_id = k, z=node[l,0]*res[0], y=node[l,1]*res[1], x=node[l,2]*res[2])
            for i in range(len(seg_list)):
                node_segment_lut[i][cc] = seg_list[i][node[l,0], node[l,1], node[l,2]]
            cc += 1
        for l in range(edge.shape[0]):
            gt_graph.add_edge(edge[l,0], edge[l,1])
    return gt_graph, node_segment_lut

def test_erl(gt_nodes, gt_edges, pred_seg, res= [30,6,6]):
    gt_graph, node_segment_lut = skelToNetworkX(gt_nodes, gt_edges, [pred_seg], res)
    scores = evaluate.expected_run_length(
                    skeletons=gt_graph,
                    skeleton_id_attribute='skeleton_id',
                    edge_length_attribute='length',
                    node_segment_lut=node_segment_lut[0],
                    skeleton_position_attributes=['z', 'y', 'x'])
    print('ERL:', scores)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('need three arguments: skel_path, seg_path resolution')
        print('example: python eval_erl.py xx.pkl yy.h5 30x6x6')
        # resolution: xyz-order
    # for nodes position: voxel or physical
    skel_path, seg_path, resolution = sys.argv[1], sys.argv[2], sys.argv[3]
    
    gt_nodes, gt_edges = pickle.load(open(skel_path, 'rb'), encoding="latin1")

    # assume node position is the voxel
    if len(sys.argv) > 4:
        for i in range(len(gt_nodes)):
            if len(gt_nodes[i])>0:
                gt_nodes[i] = gt_nodes[i] / resolution

    pred_seg = readVol(seg_path)
    resolution = [int(x) for x in resolution.split('x')]

    test_erl(gt_nodes, gt_edges, pred_seg, resolution)
