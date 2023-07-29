# step 1: compute node_id-segment lookup table from predicted segmemtation and node positions
# step 2: compute the ERL from the lookup table and the gt graph
# graph: networkx by default. To save memory for grand-challenge evaluation, we use netowrkx_lite
import numpy as np
from funlib import evaluate
from .data_io import read_vol


def compute_node_segment_lut(node_position, seg_list, data_type=np.uint32):
    """
    The function `compute_node_segment_lut` computes a lookup table that maps node positions to segment
    IDs using a list of segment arrays and node positions.

    :param node_position: A numpy array representing the positions of nodes in a 3D space. It has shape
    (N, 3), where N is the number of nodes and each row represents the x, y, and z coordinates of a node
    :param seg_list: seg_list is a list of 3D arrays representing segments. Each segment is represented
    by a 3D array where each element corresponds to a voxel in the segment. The values in the array
    represent the segment ID for each voxel
    :param data_type: The `data_type` parameter specifies the data type of the elements in the
    `node_segment_lut` array. In this case, it is set to `np.uint32`, which means that each element in
    the array will be an unsigned 32-bit integer
    :return: the variable "node_segment_lut", which is a list of numpy arrays.
    """
    node_segment_lut = [np.zeros([node_position.shape[0], 4], data_type)] * len(
        seg_list
    )
    for seg_id, seg_item in enumerate(seg_list):
        node_segment_lut[seg_id] = seg_item[
            node_position[:, 0], node_position[:, 1], node_position[:, 2]
        ]
    return node_segment_lut


def compute_node_segment_lut_low_mem(
    node_position, seg_name_list, chunk_num=1, data_type=np.uint32
):
    """
    The function `compute_node_segment_lut_low_mem` is a low memory version of a lookup table
    computation for node segments in a 3D volume.

    :param node_position: A numpy array containing the coordinates of each node. The shape of the array
    is (N, 3), where N is the number of nodes and each row represents the (z, y, x) coordinates of a
    node
    :param seg_name_list: A list of segment names, where each segment name is a string representing the
    name of a file containing segment data. The segment data is expected to be in the form of a 3D
    volume
    :param chunk_num: The parameter `chunk_num` is the number of chunks into which the volume is divided
    for reading. It is used in the `read_vol` function to specify which chunk to read, defaults to 1
    (optional)
    :param data_type: The parameter `data_type` is the data type of the array used to store the node segment
    lookup table. In this case, it is set to `np.uint32`, which means the array will store unsigned
    32-bit integers
    :return: a list of numpy arrays, where each array represents the node segment lookup table for a
    specific segment.
    """
    node_segment_lut = [np.zeros(node_position.shape[0], data_type)] * len(
        seg_name_list
    )
    for seg_id, seg_name in enumerate(seg_name_list):
        assert ".h5" in seg_name
        start_z = 0
        for chunk_id in range(chunk_num):
            seg = read_vol(seg_name, None, chunk_id, chunk_num)
            last_z = start_z + seg.shape[0]
            ind = (node_position[:, 0] >= start_z) * (node_position[:, 0] < last_z)
            pts = node_position[ind]
            node_segment_lut[seg_id][ind] = seg[
                pts[:, 0] - start_z, pts[:, 1], pts[:, 2]
            ]
            start_z = last_z
    return node_segment_lut


def compute_erl(gt_graph, node_segment_lut, merge_threshold=0):
    """
    The function `compute_erl` calculates the expected run length (ERL) scores for a given ground truth
    graph and node segment lookup table.

    :param gt_graph: The ground truth graph, which represents the true structure of the data. It is
    typically represented as a networkx graph
    :param node_segment_lut: A list of dictionaries where each dictionary represents a mapping between
    node IDs and segment IDs. Each dictionary corresponds to a different segment of the graph
    :param merge_threshold: The merge_threshold parameter is a value that determines the maximum
    distance between two nodes for them to be considered as part of the same segment. If the distance
    between two nodes is greater than the merge_threshold, they will be treated as separate segments,
    defaults to 0 (optional)
    :return: a list of scores.
    """
    scores = [None] * len(node_segment_lut)
    for lid, lut in enumerate(node_segment_lut):
        scores[lid] = evaluate.expected_run_length(
            skeletons=gt_graph,
            skeleton_id_attribute="skeleton_id",
            edge_length_attribute="length",
            node_segment_lut=lut,
            merge_thres=merge_threshold,
            skeleton_position_attributes=["z", "y", "x"],
        )
    return scores
