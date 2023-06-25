Python Library for ERL Evaluation
---

### ERL Evaluation
- Installation
```
conda env create -n erl-eval
source activate erl-eval
# install [funlib.evaluate](https://github.com/donglaiw/funlib.evaluate)
conda install -c conda-forge -c ostrokach-forge -c pkgw-forge graph-tool
pip install -r requirements.txt

# [Optional] install [kimimaro](https://github.com/seung-lab/kimimaro) for skeleton generation
# pip3 install kimimaro
```
- Example: `python eval_erl.py tmp/gt_skel.p snemi_train-labels.tif 30x6x6`
