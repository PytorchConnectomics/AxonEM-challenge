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
cd funlib.evaluate
python setup.py install
cd ..
```
- Example: `python eval_erl.py tmp/gt_skel.p snemi_train-labels.tif 30x6x6`

### Generate Skeleton
- install [kimimaro](https://github.com/seung-lab/kimimaro)
```
pip3 install kimimaro 
```
- Example: `python generate_skel.py snemi_train-labels.tif 30x6x6 1,2,3 tmp/gt_skel.p`
