import argparse
from data_io import read_pkl
from eval_erl import (
    compute_node_segment_lut,
    compute_mask_segment_id,
    compute_erl,
)
from networkx_lite import *


def test_AxonEM(gt_stats_path, pred_seg_path, num_chunk=1):
    """
    The function `test_AxonEM` takes in the paths to ground truth statistics and predicted segmentation,
    and computes the ERL (Error Rate of Length) for the predicted segmentation.

    :param gt_stats_path: The path to the ground truth statistics file. This file contains information
    about the ground truth graph (vertex in physical unit) and resolution (used to convert node position to voxel)
    :param pred_seg_path: The `pred_seg_path` parameter is the file path to the predicted segmentation.
    It is the path to a file that contains the predicted segmentation data
    :param num_chunk: The parameter `num_chunk` is an optional parameter that specifies the number of
    chunks to divide the computation into. It is used in the function `compute_node_segment_lut_low_mem`
    to divide the computation of the node segment lookup table into smaller chunks, which can help
    reduce memory usage and improve performance, defaults to 1 (optional)
    """
    # gt_graph: node position in physical unit (Nx3)
    # gt_no_bg: binary mask for non-axons
    gt_graph, gt_res, gt_no_bg = read_pkl(gt_stats_path)

    # node_segment_lut: seg id for each voxel location (N)
    # gt_graph: xyz order
    # voxel: zyx order
    node_segment_lut = compute_node_segment_lut(
        (gt_graph.nodes._nodes[:, -1:0:-1] // gt_res).astype(np.uint16),
        pred_seg_path,
        num_chunk,
    )
    mask_segment_id = compute_mask_segment_id(gt_no_bg, pred_seg_path, num_chunk)

    # https://donglaiw.github.io/paper/2021_miccai_axonEM.pdf
    scores = compute_erl(gt_graph, node_segment_lut, mask_segment_id)
    print(f"ERL for seg {pred_seg_path}: {scores[0]}")
    return scores


def get_arguments():
    """
    The function `get_arguments()` is used to parse command line arguments for the evaluation on AxonEM.
    :return: The function `get_arguments` returns the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="ERL evaluation with precomputed gt statistics"
    )
    parser.add_argument(
        "-s",
        "--seg-path",
        type=str,
        help="path to the segmentation prediction",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--gt-stats-path",
        type=str,
        help="path to ground truth skeleton statistics",
        default="",
    )
    parser.add_argument(
        "-c",
        "--num-chunk",
        type=int,
        help="number of chunks to process the volume",
        default=1,
    )
    return parser.parse_args()


if __name__ == "__main__":
    # python test_axonEM.py -s db/30um_human/axon_release/gt_16nm.h5 -g db/30um_human/axon_release/gt_16nm_skel_stats.p -c 1
    args = get_arguments()
    # compute erl
    test_AxonEM(args.gt_stats_path, args.seg_path, args.num_chunk)
