# Compute Engine burst training

[Google Compute Engine](https://cloud.google.com/compute/) gives you access to
very powerful virtual machines which you can use for as much or, more
importantly for most data science use cases, as little time as you need.

Common data science tasks that might take tens of minutes or even hours on your
personal can be completed in seconds on a machine with 64 cores and almost half
a terabyte of memory, and this can be done at costs of under $1.

If you find yourself downing too much coffee as you wait for your
[xgboost](https://github.com/dmlc/xgboost) models to train or for your
[GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
hyperparameter search to terminate, this guide is for you.

- - -

## Overview

There are different ways in which you can make use of Compute Engine resources
to make your model training and hyperparameter tuning process more efficient:

