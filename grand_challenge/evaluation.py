import numpy as np
import os
import json
from erl_wrapper.test_axonEM import test_AxonEM
from erl_wrapper.networkx_lite import *

from evalutils.evalutils import (
    DEFAULT_INPUT_PATH,
    DEFAULT_EVALUATION_OUTPUT_FILE_PATH,
    DEFAULT_GROUND_TRUTH_PATH,
)


class AxonEM:
    def __init__(self):
        self.num_chunk = 64
        self.human_gt = os.path.join(
            DEFAULT_GROUND_TRUTH_PATH, "gt_human_16nm_skel_stats_gc.p"
        )
        self.mouse_gt = os.path.join(
            DEFAULT_GROUND_TRUTH_PATH, "gt_mouse_16nm_skel_stats_gc.p"
        )
        self.human_input = os.path.join(
            DEFAULT_INPUT_PATH, "0_human_instance_seg_pred.h5"
        )
        self.mouse_input = os.path.join(
            DEFAULT_INPUT_PATH, "1_mouse_instance_seg_pred.h5"
        )

        for gt_file in [self.human_gt, self.mouse_gt]:
            if not os.path.exists(gt_file):
                raise FileNotFoundError(f"Ground truth file {gt_file} not found.")

        for input_file in [self.human_input, self.mouse_input]:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file {input_file} not found.")

        self.output_file = DEFAULT_EVALUATION_OUTPUT_FILE_PATH

    def evaluate(self):
        human_scores = test_AxonEM(
            gt_stats_path=self.human_gt,
            pred_seg_path=self.human_input,
            num_chunk=self.num_chunk,
        )
        mouse_scores = test_AxonEM(
            gt_stats_path=self.mouse_gt,
            pred_seg_path=self.mouse_input,
            num_chunk=self.num_chunk,
        )
        metrics = {
            "erl": np.mean(human_scores + mouse_scores),
            "erl_human": np.mean(human_scores),
            "erl_mouse": np.mean(mouse_scores),
            "scores_human": human_scores,
            "scores_mouse": mouse_scores,
        }

        with open(self.output_file, "w") as f:
            f.write(json.dumps(metrics))


if __name__ == "__main__":
    AxonEM().evaluate()
