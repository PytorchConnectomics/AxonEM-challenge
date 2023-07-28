import argparse
import numpy as np
import kimimaro
import networkx as nx
from data_io import read_vol, write_pkl
from networkx_lite import convert_networkx_to_lite


def skeletonize(
    labels,
    scale=4,
    const=500,
    obj_ids=None,
    dust_size=100,
    res=(32, 32, 30),
    num_thread=1,
):
    """
    The `skeletonize` function takes in a label image and returns the skeletonized version of the
    objects in the image using the Kimimaro library.

    :param labels: The input labels represent a 3D volume where each voxel is assigned a unique integer
    label. These labels typically represent different objects or regions in the volume
    :param scale: The scale parameter determines the scale at which the skeletonization is performed. It
    is used to control the level of detail in the resulting skeleton. Higher values of scale will result
    in a coarser skeleton, while lower values will result in a more detailed skeleton, defaults to 4
    (optional)
    :param const: The `const` parameter is a physical unit that determines the resolution of the
    skeletonization process. It represents the distance between two points in the skeletonized output. A
    higher value of `const` will result in a coarser skeleton, while a lower value will result in a
    finer skeleton, defaults to 500 (optional)
    :param obj_ids: The obj_ids parameter is a list of object IDs that specifies which labels in the
    input image should be processed. If obj_ids is set to None, it will default to all unique labels
    greater than 0 in the input image
    :param dust_size: The dust_size parameter specifies the minimum size (in terms of number of voxels)
    for connected components to be considered as valid objects. Connected components with fewer voxels
    than the dust_size will be skipped and not processed, defaults to 100 (optional)
    :param res: The "res" parameter specifies the resolution of the input volume data. It is a tuple of
    three values representing the voxel size in each dimension. For example, (32, 32, 30) means that the
    voxel size is 32 units in the x and y dimensions, and 30
    :param num_thread: The `num_thread` parameter specifies the number of threads to use for parallel
    processing. A value of 1 means single-threaded processing, while a value greater than 1 indicates
    multi-threaded processing. A value of 0 or less indicates that all available CPU cores should be
    used for parallel processing, defaults to 1 (optional)
    :return: The function `skeletonize` returns the result of the `kimimaro.skeletonize` function, which
    is the skeletonized version of the input labels.
    """
    if obj_ids is None:
        obj_ids = np.unique(labels)
        obj_ids = list(obj_ids[obj_ids > 0])
    return kimimaro.skeletonize(
        labels,
        teasar_params={
            "scale": scale,
            "const": const,  # physical units
            "pdrf_exponent": 4,
            "pdrf_scale": 100000,
            "soma_detection_threshold": 1100,  # physical units
            "soma_acceptance_threshold": 3500,  # physical units
            "soma_invalidation_scale": 1.0,
            "soma_invalidation_const": 300,  # physical units
            "max_paths": 50,  # default  None
        },
        object_ids=obj_ids,  # process only the specified labels
        # object_ids=[ ... ], # process only the specified labels
        # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
        # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
        dust_threshold=dust_size,  # skip connected components with fewer than this many voxels
        #       anisotropy=(30,30,30), # default True
        anisotropy=res,  # default True
        fix_branching=True,  # default True
        fix_borders=True,  # default True
        progress=True,  # default False, show progress bar
        parallel=num_thread,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=100,  # how many skeletons to process before updating progress bar
    )


def skeleton_to_networkx(
    skeletons, skeleton_resolution=None, return_all_nodes=False
):
    """
    The function `skeleton_to_networkx` converts a skeleton object into a networkx graph, with an option
    to return all nodes.

    :param skeletons: The "skeletons" parameter is a list of skeleton objects. Each skeleton object
    represents a graph structure with nodes and edges. The function converts these skeleton objects into
    a networkx graph object
    :param skeleton_resolution: The `skeleton_resolution` parameter is an optional parameter that
    specifies the resolution of the skeleton. It is used to scale the node coordinates in the skeleton.
    If provided, the node coordinates will be multiplied by the skeleton resolution
    :param return_all_nodes: The `return_all_nodes` parameter is a boolean flag that determines whether
    or not to return all the nodes in the graph. If `return_all_nodes` is set to `True`, the function
    will return both the graph object and an array of all the nodes in the graph. If `return_all,
    defaults to False (optional)
    :return: The function `skeleton_to_networkx` returns a networkx graph object representing the
    skeleton. Additionally, if the `return_all_nodes` parameter is set to `True`, the function also
    returns an array of all the nodes in the skeleton.
    """

    # node in gt_graph: physical unit
    gt_graph = nx.Graph()
    count = 0
    all_nodes = [None] * len(skeletons)
    for skeleton_id, skeleton in enumerate(skeletons):
        if len(skeleton.edges) == 0:
            continue
        node_arr = skeleton.vertices.astype(np.uint16)
        if skeleton_resolution is not None:
            node_arr = node_arr * skeleton_resolution
        # augment the node index
        edge_arr = skeleton.edges + count
        for node in node_arr:
            # unit: physical
            gt_graph.add_node(
                count, skeleton_id=skeleton_id, z=node[0], y=node[1], x=node[2]
            )
            count += 1
        for edge in edge_arr:
            gt_graph.add_edge(edge[0], edge[1])
        if return_all_nodes:
            all_nodes[skeleton_id] = node_arr

    if return_all_nodes:
        all_nodes = np.vstack(all_nodes)
        return gt_graph, all_nodes
    return gt_graph


def node_edge_to_networkx(
    nodes, edges, skeleton_resolution=None, return_all_nodes=False
):
    """
    The function `node_edge_to_networkx` converts a set of nodes and edges into a networkx graph object,
    optionally returning all nodes as well.

    :param nodes: A list of arrays, where each array represents the coordinates of nodes in a skeleton.
    Each array corresponds to a different skeleton
    :param edges: The `edges` parameter is a list of lists, where each inner list represents an edge in
    the graph. Each inner list contains two elements, which are the indices of the nodes that the edge
    connects
    :param skeleton_resolution: The `skeleton_resolution` parameter is an optional argument that
    specifies the resolution of the skeleton. It is used to scale the node coordinates in the `node_arr`
    array. If `skeleton_resolution` is provided, the node coordinates are multiplied by the resolution
    value. This is useful when working with
    :param return_all_nodes: The `return_all_nodes` parameter is a boolean flag that determines whether
    or not to return all the nodes in the graph. If set to `True`, the function will return both the
    graph and an array containing all the nodes. If set to `False` (default), the function will only
    return, defaults to False (optional)
    :return: a networkx graph object. If the parameter `return_all_nodes` is set to `True`, it also
    returns an array of all the nodes in the graph.
    """

    gt_graph = nx.Graph()
    count = 0
    all_nodes = [None] * len(nodes)
    for skeleton_id, node_arr in enumerate(nodes):
        if len(edges[skeleton_id]) == 0:
            continue
        node_arr = node_arr.astype(np.uint16)
        if skeleton_resolution is not None:
            node_arr = node_arr * skeleton_resolution
        # augment the node index
        edge_arr = edges[skeleton_id] + count
        for node in node_arr:
            # unit: physical
            gt_graph.add_node(
                count, skeleton_id=skeleton_id, z=node[0], y=node[1], x=node[2]
            )
            count += 1
        for edge in edge_arr:
            gt_graph.add_edge(edge[0], edge[1])
        if return_all_nodes:
            all_nodes[skeleton_id] = node_arr
    if return_all_nodes:
        all_nodes = np.vstack(all_nodes)
        return gt_graph, all_nodes
    return gt_graph


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Generate skeleton results from input segmentation"
    )
    parser.add_argument(
        "-s",
        "--seg-path",
        type=str,
        help="path to the segmentation",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--seg-resolution",
        type=str,
        help="segmentation resolution (zyx order)",
        default="30,6,6",
    )
    parser.add_argument(
        "-d",
        "--dust-size",
        type=int,
        help="dust size parameter for skeletonization",
        default=100,
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="output path",
        default="out.pkl",
    )
    parser.add_argument(
        "-i",
        "--seg-index",
        type=str,
        help=(
            "selected segmentation indices for skeletonization. '-1' means all"
        ),
        default="-1",
    )

    result_args = parser.parse_args()

    # parse segmentation resolution
    result_args.seg_resolution = [
        int(x) for x in result_args.seg_resolution.split("x")
    ]
    # parse skeleton index
    result_args.seg_index = (
        None
        if result_args.seg_index == "-1"
        else [int(x) for x in result_args.seg_index.split(",")]
    )
    return result_args


if __name__ == "__main__":
    args = get_arguments()
    # python skeleton.py yy.h5 30x6x6 -1 xx.pkl
    print("load segmentation")
    seg = read_vol(args.seg_path)

    print("start skeletonization")
    result_skeletons = skeletonize(
        seg,
        obj_ids=args.seg_index,
        dust_size=args.dust_size,
        res=args.seg_resolution,
    )
    print("save output")
    if args.output_type == "skeleton":
        write_pkl(args.output_path, result_skeletons)
    elif args.output_type == "networkx":
        result_networkx = skeleton_to_networkx(result_skeletons)
        write_pkl(args.output_path, result_networkx)
    elif args.output_type == "erl":
        # for erl evaluation
        result_networkx, result_all_nodes = skeleton_to_networkx(
            result_skeletons, True
        )
        result_all_nodes_voxel = result_all_nodes // args.seg_resolution
        result_networkx_lite = convert_networkx_to_lite(result_networkx)
        write_pkl(
            args.output_path, [result_networkx_lite, result_all_nodes_voxel]
        )
