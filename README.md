# 1D_xrd_ml
Repo for Carlson summer project: designing an ML tool for 1D XRD scattering data

# Docker: Automatic detection

## Description
This docker container automatically finds the peaks within a group of 1D XRD data files.

To do this create the following directories:

Input file directory: automatic_detection/data/input_data

Output file directory: automatic_detection/data/results

The input file directory should contain the XRD files.

## Running
First, let's create the image:
```
cd automatic_detection
make build_docker
```
Then execute:
```
make run_docker
```

Notes:
- It is assumed that splash-ml handles the path to the dataset, such that file can be read, e.g. "C:/Users/..." vs "/mnt/c/Users/..."
