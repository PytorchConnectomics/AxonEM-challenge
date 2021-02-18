Useful Scripts
---

### Generate Skeleton
- install [kimimaro](https://github.com/seung-lab/kimimaro)
```
pip3 install kimimaro 
```

- Example: `python generate_skel.py snemi_train-labels.tif 30x6x6 1,2,3 tmp/gt_skel.p`

### ERL Evaluation
- install [funlib.evaluate](https://github.com/funkelab/funlib.evaluate)
```
conda env create -n erl-eval
source activate erl-eval
conda install -c conda-forge -c ostrokach-forge -c pkgw-forge graph-tool
pip install -r requirements.txt
pip install --editable .
```
- Example: `python test_skel.py tmp/gt_skel.p snemi_train-labels.tif 30x6x6`
