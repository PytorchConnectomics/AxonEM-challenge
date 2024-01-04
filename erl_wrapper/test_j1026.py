import argparse
import os
from data_io import read_pkl, mkdir, read_vol, write_vol, write_pkl
from eval_erl import (
    compute_segment_lut,
    compute_erl,
    compute_segment_lut_tile,
    compute_segment_lut_tile_combine,
)
from skeleton import node_edge_to_networkx
from networkx_lite import convert_networkx_to_lite
    

def get_file_path(folder, name):
    if name == 'gt_vertices':
        return os.path.join(folder, 'gt_vertices.h5')
    elif name == 'gt_graph':
       return os.path.join(folder, 'gt_graph.pkl')
    elif name == 'seg_pred'
       return os.path.join(folder, "%04d", "%d_%d.h5")
    elif name == 'seg_lut':
        return os.path.join(folder, "%04d", "%d_%d.h5")
    elif name == 'seg_lut_all':
        return os.path.join(folder, "seg_lut_all.h5")    
    raise f"File not found: {name}"

def compute_lut_j0126(output_folder, option, seg_folder="", gt_skeleton="", job=[0, 1]):
    seg_lut_path = get_file_path(output_folder, 'seg_lut') 
    zran = 128 * np.arange(45)
    yran = np.arange(6)
    xran = np.arange(6)
    if option == "map":
        seg_pred_path = get_file_path(seg_folder, 'seg_pred')        
        mkdir(output_folder)
        zran = zran[job[0] :: job[1]]
        pts = read_vol(gt_skeleton)
        compute_segment_lut_tile(
            seg_pred_path, zran, yran, xran, pts, seg_lut_path
        )
    elif option == "reduce":
        seg_lut_all_path = get_file_path(output_folder, 'seg_lut_all')
        if os.path.exists(seg_lut_all_path):
            print(f"File exists: ${seg_lut_path}")
        else:
            # check that all files exist
            for z in zran:
                for y in yran:
                    for x in xran:
                        sn = seg_lut_path % (z, y, x)
                        if not os.path.exists(sn):
                            raise f"File not exists: {sn}"

            out = compute_segment_lut_tile_combine(zran, yran, xran, seg_lut_path)
            write_vol(seg_lut_all_path, out)        

def compute_erl_j0126():
    print("Load gt info")
    # convert pts into graph
    # gt_no_bg: binary mask for non-axons        

    print("Compute prediction info")
    # node_segment_lut: seg id for each voxel location (N)
    # gt_graph: xyz order
    # voxel: zyx order
     scores = compute_erl(
        gt_graph, node_segment_lut, mask_segment_id, merge_threshold, erl_intervals
    )

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
        "-t",
        "--task",
        type=int,
        help="0: compute the segment id for each gt skeleton point, 1: compute erl",
        default=0,
    )
    parser.add_argument(
        "-s",
        "--seg-folder",
        type=str,
        help="path to FFN segmentation prediction folder",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--gt-skeleton",
        type=str,
        help="path to ground truth skeleton file",
        default="",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        help="path to output evaluation",
        default="eval",
    )
    parser.add_argument(
        "-j",
        "--job",
        type=str,
        help="job_id,job_num: compute task=0 in parallel",
        default="0,1",
    )
    parser.add_argument(
        "-mt",
        "--merge-threshold",
        type=int,
        help="threshold number of voxels to be a false merge",
        default=50,
    )

    args = parser.parse_args()

    if len(args.gt_mask_path) == 0:
        args.gt_mask_path = None
    args.job = [int(x) for x in args.job.split(",")]
    return args

def compute_skeleton_j0126(gt_skeleton, output_folder:
    skeletons = read_vol(gt_skeleton)    
    vertices_path = get_file_path(output_folder, 'gt_vertices')
    graph_path = get_file_path(output_folder, 'gt_graph') 
    
    if not os.path.exists(vertices_path) or os.path.exists(graph_path):
        vertices = [np.array(skeletons[k]['vertices']). astype(np.uint16) for k in skeletons.keys()] 
        if not os.path.exists(vertices_path): 
            write_vol(pts_path, np.vstack(vertices))
        if not os.path.exists(graph_path):
            edges = [np.array(skeletons[k]['edges']). astype(np.uint16) for k in skeletons.keys()]                     
            gt_graph = node_edge_to_networkx(vertices, edges, [20,10,10])
            gt_graph_lite = convert_networkx_to_lite(gt_graph)
            write_pkl(graph_path, gt_graph_lite)    

        
if __name__ == "__main__":
    # python test_axonEM.py -s db/30um_human/axon_release/gt_16nm.h5 -g db/30um_human/axon_release/gt_16nm_skel_stats.p -c 1
    args = get_arguments()

    if args.task == 0:
        print("Step 0: process the gt skeleton pts")
        compute_skeleton_j0126(args.gt_skeleton, args.output_folder)                
    elif args.task == 1:
        print("Step 1: compute segment id for each seg tile")
        compute_lut_j0126(
            args.output_folder, "map", args.seg_folder, get_file_path(args, 'gt_vertices'), args.job
        )
    elif args.task == 2:
        print("Step 2: combine segment id results for all seg tiles')
        compute_lut_j0126(args.output_folder, "reduce")        
    elif args.task == 3: 
        print("Step 3: compute erl")
        gt_graph = read_pkl(get_file_path(output_folder, 'gt_graph'))
        node_segment_lut = read_vol(get_file_path(output_folder, 'seg_lut_all'))
        scores = compute_erl(gt_graph, node_segment_lut)
        print(f"ERL/GT for seg {args.seg_folder}: {scores}")
