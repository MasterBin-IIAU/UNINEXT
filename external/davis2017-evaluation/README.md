# DAVIS 2017 Semi-supervised and Unsupervised evaluation package

This package is used to evaluate semi-supervised and unsupervised video multi-object segmentation models for the <a href="https://davischallenge.org/davis2017/code.html" target="_blank">DAVIS 2017</a> dataset. 

This tool is also used to evaluate the submissions in the Codalab site for the <a href="https://competitions.codalab.org/competitions/20516" target="_blank">Semi-supervised DAVIS Challenge</a> and the <a href="https://competitions.codalab.org/competitions/20515" target="_blank">Unsupervised DAVIS Challenge</a>

### Installation
```bash
# Download the code
git clone https://github.com/davisvideochallenge/davis2017-evaluation.git && cd davis2017-evaluation
# Install it - Python 3.6 or higher required
python setup.py install
```
If you don't want to specify the DAVIS path every time, you can modify the default value in the variable `default_davis_path` in `evaluation_method.py`(the following examples assume that you have set it). 
Otherwise, you can specify the path in every call using using the flag `--davis_path /path/to/DAVIS` when calling `evaluation_method.py`.

Once the evaluation has finished, two different CSV files will be generated inside the folder with the results: 
- `global_results-SUBSET.csv` contains the overall results for a certain `SUBSET`. 
- `per-sequence_results-SUBSET.csv` contain the per sequence results for a certain `SUBSET`.

If a folder that contains the previous files is evaluated again, the results will be read from the CSV files instead of recomputing them.

## Evaluate DAVIS 2017 Semi-supervised
In order to evaluate your semi-supervised method in DAVIS 2017, execute the following command substituting `results/semi-supervised/osvos` by the folder path that contains your results:
```bash
python evaluation_method.py --task semi-supervised --results_path results/semi-supervised/osvos
```
The semi-supervised results have been generated using [OSVOS](https://github.com/kmaninis/OSVOS-caffe).

## Evaluate DAVIS 2017 Unsupervised
In order to evaluate your unsupervised method in DAVIS 2017, execute the following command substituting `results/unsupervised/rvos` by the folder path that contains your results:
```bash
python evaluation_method.py --task unsupervised --results_path results/unsupervised/rvos
```
The unsupervised results example have been generated using [RVOS](https://github.com/imatge-upc/rvos).

## Evaluation running in Codalab
In case you would like to know which is the evaluation script that is running in the Codalab servers, check the `evaluation_codalab.py` script.

This package runs in the following docker image: [scaelles/codalab:anaconda3-2018.12](https://cloud.docker.com/u/scaelles/repository/docker/scaelles/codalab)

## Citation

Please cite both papers in your publications if DAVIS or this code helps your research.

```latex
@article{Caelles_arXiv_2019,
  author = {Sergi Caelles and Jordi Pont-Tuset and Federico Perazzi and Alberto Montes and Kevis-Kokitsi Maninis and Luc {Van Gool}},
  title = {The 2019 DAVIS Challenge on VOS: Unsupervised Multi-Object Segmentation},
  journal = {arXiv},
  year = {2019}
}
```

```latex
@article{Pont-Tuset_arXiv_2017,
  author = {Jordi Pont-Tuset and Federico Perazzi and Sergi Caelles and Pablo Arbel\'aez and Alexander Sorkine-Hornung and Luc {Van Gool}},
  title = {The 2017 DAVIS Challenge on Video Object Segmentation},
  journal = {arXiv:1704.00675},
  year = {2017}
}
```

