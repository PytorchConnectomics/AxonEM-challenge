import argparse
import numpy as np
from data_io import read_vol, read_pkl
from skeleton import skeleton_to_networkx
from eval_erl import compute_node_segment_lut, compute_erl


def test_volume(
    seg_path,
    skeleton_path,
    skeleton_unit,
    skeleton_resolution,
):
    """
    The function `test_volume` takes in various inputs, including a segmentation path, a skeleton path,
    and parameters related to units, and performs computations to calculate the ERL
    (Error Rate of Length) for the given segmentation.

    :param seg_path: The path to the segmentation volume file
    :param skeleton_path: The `skeleton_path` parameter is the path to the file that contains the
    skeleton data
    :param skeleton_unit: The parameter "skeleton_unit" determines the unit of measurement for the
    skeleton. It can have two possible values: "physical" or "voxel"
    :param skeleton_resolution: The `skeleton_resolution` parameter represents the voxel size of the
    skeleton data. It is used to convert the node positions from physical units to voxel units if
    `skeleton_unit` is set to "voxel"
    """

    pred_seg = read_vol(seg_path)
    gt_skeleton = read_pkl(skeleton_path)

    # graph: need physical unit
    # node position: need voxel unit
    if skeleton_unit == "physical":
        gt_graph, all_nodes = skeleton_to_networkx(gt_skeleton, None, True)
        all_nodes = all_nodes // skeleton_resolution
    else:
        gt_graph, all_nodes = skeleton_to_networkx(
            gt_skeleton, skeleton_resolution, True
        )

    node_segment_lut = compute_node_segment_lut(all_nodes, [pred_seg])
    scores = compute_erl(gt_graph, node_segment_lut)
    print(f"ERL for seg {seg_path}: {scores[0]}")


def get_arguments():
    """
    The `get_arguments` function is used to parse command line arguments for the ERL evaluation on small
    volume.
    :return: the parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser(description="ERL evaluation on small volume")
    parser.add_argument(
        "-s",
        "--seg-path",
        type=str,
        help="path to the segmentation prediction",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--skeleton-path",
        type=str,
        help="path to ground truth skeleton",
        default="",
    )
    parser.add_argument(
        "-gu",
        "--skeleton-unit",
        type=str,
        choices=["physical", "voxel"],
        help="unit of the skeleton node positions",
        default="voxel",
    )
    parser.add_argument(
        "-gr",
        "--skeleton-resolution",
        type=str,
        help="resolution of ground truth skeleton",
        required=True,
    )
    result_args = parser.parse_args()
    assert (
        "x" in result_args.skeleton_resolution
    ), "The gt skeleton resolution needs to be in the format of axbxc"
    result_args.skeleton_resolution = [
        int(x) for x in result_args.skeleton_resolution.split("x")
    ]
    return result_args


if __name__ == "__main__":
    args = get_arguments()
    test_volume(
        args.seg_path,
        args.skeleton_path,
        args.skeleton_unit,
        args.skeleton_resolution
    )
