# Evaluation

This folder contains the evaluation code for generic competitors (including our solution)

## Example usage:

Run predict `our` on `faust` and `faust_permuted`:
```bash
PYTHONPATH=. python evaluation/predict.py -m dataset=faust,faust_permuted model._target_=evaluation.competitors.our.our.OurMatching
```

Then, run the evaluation of those predictions:
```bash
PYTHONPATH=. python evaluation/evaluation.py -m dataset=faust,faust_permuted model._target_=evaluation.competitors.our.our.OurMatching
```

With the default `conf`, the grid search is parallel

## Scripts

The main scripts are:

- `predict.py`: predict the matchings of some competitor against some dataset
- `evaluate.py`: evaluate the previously predicted matchings
- `print_performance.py`: pretty print the `mean_geo_error` of all methods/dataset

The `predict.py` and `evaluate.py` configuration is defined in the `conf` folder, with
a simple `yml` file.


## Download data

To download all the data (datasets, prediction, performance) it is enough to run:

```bash
dvc pull
```

The data described by the hashes in the `*.dvc` files will be checked-out.



## File tree structure

The evaluation folder is organized as follow:

```bash
.
├── conf          # general hydra configuration, enables cli parallel multiruns
├── competitors   # competitors implementations/wrappers 
├── datasets      # the datasets available
├── performance   # the evaluation of the predictions
└── predictions   # the prediction of some method on some dataset
```
