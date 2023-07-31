Python Library for ERL Evaluation
---

### ERL Evaluation
- Installation
```
# create a new environment
conda create -n erl-eval python==3.9.0
source activate erl-eval
conda install -c conda-forge -c ostrokach-forge -c pkgw-forge graph-tool

# clone the repo
git clone --recursive https://github.com/PytorchConnectomics/AxonEM-challenge.git

# install [funlib.evaluate](https://github.com/donglaiw/funlib.evaluate)
cd challenge_eval/funlib.evaluate
pip install -r requirements.txt
python setup.py install
cd ..
```
(Under `challenge_eval/` folder)
- AxonEM evaluation: `python test_axonEM.py -s seg_axonM.h5 -g axonM_gt_16nm_skel_stats.p -c 5`

### Generate Skeleton
- install [kimimaro](https://github.com/seung-lab/kimimaro)
```
pip3 install kimimaro 
```
(Under `challenge_eval/` folder)
- GT skeleton generation: `python skeleton.py -s snemi_train-labels.tif -r 30x6x6 -i 1,2,3 -o snemi_skel.p`
- ERL evaluation: `python test_volume.py -s pred_seg.tif -g snemi_skel.p -gu physical -gr 30x6x6`
