# Evaluation datasets

This folder contains one folder for each evaluation dataset,
the name of each folder must uniquely identify the corresponding datasets.


## Download datasets

To download the datasets it is enough to run:

```bash
dvc pull
```


## File tree structure

Each dataset folder is organized as follows:

```bash
.
├── data
│   ├── 000                         # sample index
│   │   ├── A.off                   # shape A in .off format
│   │   ├── B.off                   # shape B in .off format
│   │   ├── gt_matching_A_to_B.txt  # ground truth matching from A to B
│   │   └── meta.json               # meta information, e.g. shapes original names
│   └── ...
├── generate.py                     # script used to generate this dataset
└── README.md                       # general info about this dataset
```

The file `gt_matching_A_to_B.txt` contains the ground truth matching.
The index of each row refers to a vertex in the shape `A`,
the value in that row identifies the corresponding vertex in `B`.
