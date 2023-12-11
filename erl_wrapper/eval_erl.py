# step 1: compute node_id-segment lookup table from predicted segmemtation and node positions
# step 2: compute the ERL from the lookup table and the gt graph
# graph: networkx by default. To save memory for grand-challenge evaluation, we use netowrkx_lite
import numpy as np
from funlib import evaluate
from data_io import read_vol


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


def compute_erl(gt_graph, node_segment_lut):
    """
    The function `compute_erl` calculates the expected run length (ERL) scores for a given ground truth
    graph and node segment lookup table.

    :param gt_graph: The ground truth graph, which represents the true structure of the data. It is
    typically represented as a networkx graph
    :param node_segment_lut: A list of dictionaries where each dictionary represents a mapping between
    node IDs and segment IDs. Each dictionary corresponds to a different segment of the graph
    :return: a list of scores.
    """
    scores = [None] * len(node_segment_lut)
    for lid, lut in enumerate(node_segment_lut):
        scores[lid] = expected_run_length(
            skeletons=gt_graph,
            skeleton_id_attribute="skeleton_id",
            edge_length_attribute="length",
            node_segment_lut=lut,
            skeleton_position_attributes=["z", "y", "x"],
        )
    return scores

# copy from https://github.com/funkelab/funlib.evaluate/blob/master/funlib/evaluate/run_length.py
def expected_run_length(
        skeletons,
        skeleton_id_attribute,
        edge_length_attribute,
        node_segment_lut,
        skeleton_lengths=None,
        skeleton_position_attributes=None,
        return_merge_split_stats=False):
    '''Compute the expected run-length on skeletons, given a segmentation in
    the form of a node -> segment lookup table.

    Args:

        skeletons:

            A networkx-like graph.

        skeleton_id_attribute:

            The name of the node attribute containing the skeleton ID.

        edge_length_attribute:

            The name of the edge attribute for the length of an edge. If
            `skeleton_position_attributes` is given, the lengthes will be
            computed from the node positions and stored in this attribute. If
            `skeleton_position_attributes` is not given, it is assumed that the
            lengths are already stored in `edge_length_attribute`.

            The latter use case allows to precompute the the edge lengths using
            function `get_skeleton_lengths` below, and reuse them for
            subsequent calls with different `node_segment_lut`s.

        node_segment_lut:

            A dictionary mapping node IDs to segment IDs.

        skeleton_lengths (optional):

            A dictionary from skeleton IDs to their length. Has to be given if
            precomputed edge lengths are to be used (see argument
            `edge_length_attribute`).

        skeleton_position_attributes (optional):

            A list of strings with the names of the node attributes for the
            spatial coordinates.

        return_merge_split_stats (optional):

            If ``True``, return a dictionary with additional split/merge stats
            together with the run length, i.e., ``(run_length, stats)``.

            The merge stats are a dictionary mapping segment IDs to a list of
            skeleton IDs that got merged.

            The split stats are a dictionary mapping skeleton IDs to pairs of
            segment IDs, one pair for each split along the skeleton edges.
    '''

    if skeleton_position_attributes is not None:

        if skeleton_lengths is not None:
            raise ValueError(
                "If skeleton_position_attributes is given, skeleton_lengths"
                "should not be given")

        skeleton_lengths = get_skeleton_lengths(
            skeletons,
            skeleton_position_attributes,
            skeleton_id_attribute,
            store_edge_length=edge_length_attribute)

    total_skeletons_length = np.sum([l for _, l in skeleton_lengths.items()])
    res = evaluate_skeletons(
        skeletons,
        skeleton_id_attribute,
        node_segment_lut,
        return_merge_split_stats=return_merge_split_stats)

    if return_merge_split_stats:
        skeleton_scores, merge_split_stats = res
    else:
        skeleton_scores = res

    skeletons_erl = 0
    db = {}
    db2 = {}
    db3 = {}

    for skeleton_id, scores in skeleton_scores.items():

        skeleton_length = skeleton_lengths[skeleton_id]
        skeleton_erl = 0

        for segment_id, correct_edges in scores.correct_edges.items():

            correct_edges_length = np.sum([
                skeletons.edges[e][edge_length_attribute]
                for e in correct_edges])

            skeleton_erl += (
                correct_edges_length *
                (correct_edges_length/skeleton_length)
            )
        skeletons_erl += (
            (skeleton_length/total_skeletons_length) *
            skeleton_erl
        )
        db[skeleton_id] = skeleton_erl
        db2[skeleton_id] = skeleton_length
        db3[skeleton_id] = correct_edges_length

    import pdb; pdb.set_trace()
    # db3 = np.array(list(db3.values())); db2 = np.array(list(db2.values())); db = np.array(list(db.values()))
    if return_merge_split_stats:
        return skeletons_erl, merge_split_stats
    else:
        return skeletons_erl


def get_skeleton_lengths(
        skeletons,
        skeleton_position_attributes,
        skeleton_id_attribute,
        store_edge_length=None):
    '''Get the length of each skeleton in the given graph.

    Args:

        skeletons:

            A networkx-like graph.

        skeleton_position_attributes:

            A list of strings with the names of the node attributes for the
            spatial coordinates.

        skeleton_id_attribute:

            The name of the node attribute containing the skeleton ID.

        store_edge_length (optional):

            If given, stores the length of an edge in this edge attribute.
    '''

    node_positions = {
        node: np.array(
            [
                skeletons.nodes[node][d]
                for d in skeleton_position_attributes
            ],
            dtype=np.float32)
        for node in skeletons.nodes()
    }

    skeleton_lengths = {}
    for u, v, data in skeletons.edges(data=True):

        skeleton_id = skeletons.nodes[u][skeleton_id_attribute]

        if skeleton_id not in skeleton_lengths:
            skeleton_lengths[skeleton_id] = 0

        pos_u = node_positions[u]
        pos_v = node_positions[v]

        length = np.linalg.norm(pos_u - pos_v)

        if store_edge_length:
            data[store_edge_length] = length
        skeleton_lengths[skeleton_id] += length

    return skeleton_lengths


class SkeletonScores():

    def __init__(self):

        self.ommitted = 0
        self.split = 0
        self.merged = 0
        self.correct = 0
        self.correct_edges = {}


def evaluate_skeletons(
        skeletons,
        skeleton_id_attribute,
        node_segment_lut,
        return_merge_split_stats=False):

    # find all merging segments (skeleton edges on merging segments will be
    # counted as wrong)

    # pairs of (skeleton, segment), one for each node
    skeleton_segment = np.array([
        [data[skeleton_id_attribute], node_segment_lut[n]]
        for n, data in skeletons.nodes(data=True)
    ])

    # unique pairs of (skeleton, segment)
    skeleton_segment, count = np.unique(skeleton_segment, axis=0, return_counts=True)

    # number of times that a segment was mapped to a skeleton
    segments, num_segment_skeletons = np.unique(
        skeleton_segment[:, 1],
        return_counts=True)

    # all segments that merge at least two skeletons
    merging_segments = segments[num_segment_skeletons > 1]

    merging_segments_mask = np.isin(skeleton_segment[:, 1], merging_segments)
    merged_skeletons = skeleton_segment[:, 0][merging_segments_mask]
    merging_segments = set(merging_segments)
    import pdb; pdb.set_trace()
    # skeleton_segment[skeleton_segment[:,1]==207]

    merges = {}
    splits = {}

    if return_merge_split_stats:

        merged_segments = skeleton_segment[:, 1][merging_segments_mask]

        for segment, skeleton in zip(merged_segments, merged_skeletons):
            if segment not in merges:
                merges[segment] = []
            merges[segment].append(skeleton)

    merged_skeletons = set(np.unique(merged_skeletons))

    skeleton_scores = {}

    for u, v in skeletons.edges():

        skeleton_id = skeletons.nodes[u][skeleton_id_attribute]
        segment_u = node_segment_lut[u]
        segment_v = node_segment_lut[v]

        if skeleton_id not in skeleton_scores:
            scores = SkeletonScores()
            skeleton_scores[skeleton_id] = scores
        else:
            scores = skeleton_scores[skeleton_id]

        if segment_u == 0 or segment_v == 0:
            scores.ommitted += 1
            continue

        if segment_u != segment_v:
            scores.split += 1
            if return_merge_split_stats:
                if skeleton_id not in splits:
                    splits[skeleton_id] = []
                splits[skeleton_id].append((segment_u, segment_v))
            continue

        # segment_u == segment_v != 0
        segment = segment_u

        # potentially merged edge?
        if skeleton_id in merged_skeletons:
            if segment in merging_segments:
                scores.merged += 1
                continue

        scores.correct += 1

        if segment not in scores.correct_edges:
            scores.correct_edges[segment] = []
        scores.correct_edges[segment].append((u, v))

    if return_merge_split_stats:

        merge_split_stats = {
            'merge_stats': merges,
            'split_stats': splits
        }

        return skeleton_scores, merge_split_stats

    else:

        return skeleton_scores