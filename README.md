# RANSAC

## Overview
This is an implementation of Random Sample Consensus used to remove outliers from a data set.

## Setup

Install packages NumPy, SciPy, and matplotlib.pyplot before running the experiment.

## Testing
To test the model, navigate to the repository in your terminal and type

```bash
python experiment.py
```

The model finds inliers by choosing two random points, computes a line, and investigates which points are close to it. This is done iteratively, as seen in figure 1. The line with the most points close to it wins and these points are returned as inliers.

<img src="https://media.giphy.com/media/HlKD5zj41UMUccs4zt/giphy.gif" width="470" height="350">
Figure 1. A demonstration of RANSAC finding the optimal parameters. Red points show inliers, and blue points show outliers. 
