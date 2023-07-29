import numpy as np
import os
import json
from challenge_eval.test_axonEM import test_AxonEM
from challenge_eval.networkx_lite import *

from evalutils.evalutils import (
    DEFAULT_INPUT_PATH,
    DEFAULT_EVALUATION_OUTPUT_FILE_PATH,
    DEFAULT_GROUND_TRUTH_PATH,
)


class AxonEM:
    def __init__(self):
        self.gt_file = os.path.join(DEFAULT_GROUND_TRUTH_PATH, "test-labels.p")
        self.input_file = os.path.join(DEFAULT_INPUT_PATH, "test-input.h5")
        self.output_file = DEFAULT_EVALUATION_OUTPUT_FILE_PATH

    def evaluate(self):
        scores = test_AxonEM(
            gt_stats_path=self.gt_file, pred_seg_path=self.input_file, num_chunk=64
        )
        metrics = {"erl": scores[0], "scores": scores}

        with open(self.output_file, "w") as f:
            f.write(json.dumps(metrics))


if __name__ == "__main__":
    AxonEM().evaluate()
