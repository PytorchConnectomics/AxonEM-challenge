# Grand Challenge evaluation
## Expected directory structure
```
.
├── challenge_eval
│   └── ...
├── grand_challenge
│   └── ...
├── ground-truth
│   ├── gt_human_16nm_skel_stats_gc.p
│   └── gt_mouse_16nm_skel_stats_gc.p
└── test
    ├── 0_human_instance_seg_pred.h5
    └── 1_mouse_instance_seg_pred.h5
 ```

 Might need to change permissions on `test` because of [SELinux](https://stackoverflow.com/a/24334000/10702372).
