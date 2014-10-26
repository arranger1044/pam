#K-Medoids

Simple implementation of the **Partitioning Around Medoids** (_**PAM**_) using numba or theano to speed up the computation.

##Basic usage

Here's a straightforward example of how to call it from the shell:
```
    ipython -- pampy.py path_to_csv_data -i 1000 -k 5 --theano --cluster-ids
```

where:

* `path_to_csv_data` is the path to your data in csv format
* `-i` specifies the number of iterations until convergence
* `-k` specities the number of clusters to find
* `--theano` tells whether to use theano or not (numba implementation is used instead)
* `--cluster-ids` tells whether the final clustering shall be returned with medoid ids or progressive ids as labels
