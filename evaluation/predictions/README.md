# Predictions

This folder contains the predicted (and ground truth for sanity checks and convenience),
matching, from A to B.


## Structure

```bash
.
├── faust                                   # The dataset considered
│   └── euclidean                           # The model considered
│       └── 000                             # The sample id in that dataset
│           ├── gt_matching_A_to_B.txt      # The ground truth matching
│           └── pred_matching_A_to_B.txt    # The predicted matching
├── ...
```


The file `gt_matching_A_to_B.txt` contains the ground truth matching,
the file `pred_matching_A_to_B.txt` contains the predicted truth matching.

In these files the index of each row refers to a vertex in the shape `A`,
the value in that row identifies the corresponding vertex in `B`.
