import argparse
import numpy as np
from data_io import read_vol, read_pkl
from skeleton import skeleton_to_networkx
from eval_erl import *
import pickle


def test_volume(
    seg_path,
    skeleton_path,
    skeleton_unit,
    skeleton_resolution,
    erl_merge_threshold,
):
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
    scores = compute_erl(gt_graph, node_segment_lut, erl_merge_threshold)
    print(f"ERL for seg {seg_path}: {scores[0]}")


def get_arguments():
    """
    The `get_arguments` function is used to parse command line arguments for the ERL evaluation on small
    volume.
    :return: the parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser(
        description="ERL evaluation on small volume"
    )
    parser.add_argument(
        "-sg",
        "--seg-path",
        type=str,
        help="path to the segmentation prediction",
        required=True,
    )
    parser.add_argument(
        "-sk",
        "--skeleton-path",
        type=str,
        help="path to ground truth skeleton",
        default="",
    )
    parser.add_argument(
        "-u",
        "--skeleton-unit",
        type=str,
        choices=["physical", "voxel"],
        help="unit of the skeleton node positions",
        default="voxel",
    )
    parser.add_argument(
        "-r",
        "--skeleton-resolution",
        type=str,
        help="resolution of ground truth skeleton",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--erl-merge-threshold",
        type=int,
        help="merge threshold for erl evaluation",
        default=0,
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
        args.skeleton_resolution,
        args.erl_merge_threshold,
    )
