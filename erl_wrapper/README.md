Reproducing FFN evaluation result (test_j0126.py)
----
[paper](https://www.nature.com/articles/s41592-018-0049-4), [data from the paper](https://storage.googleapis.com/j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/README.txt)
- Data download (our processed version)
    = GT skeleton [test (50 neurons)](https://rhoana.rc.fas.harvard.edu/dataset/j0126/test_50_skeletons.h5), [validation (12 neurons)](https://rhoana.rc.fas.harvard.edu/dataset/j0126/valid_12_skeletons.h5)
    = FFN segmentation: [whole
    = (optional) training data: [33 subvolumes](https://rhoana.rc.fas.harvard.edu/dataset/j0126/train.zip)
- Run script
```
# unzip FFN segmentation: XX/ffn_agg_20-10-10/
# gt skeleton file: YY/test_50_skeletons.h5
# step 1: compute the segment id for each gt skeleton point (for parallellism, use `--job ${job_id},${job_num}`) 
python test_j0126.py --task 0 --seg-folder XX/ffn_agg_20-10-10/ --gt-skeleton YY/test_50_skeletons.h5
# step 2: compute the segment id for each gt skeleton point 
python test_j0126.py -t 1 
```
