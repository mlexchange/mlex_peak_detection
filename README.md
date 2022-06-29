# Peak Detection in 1-D XRD data

## Description
This docker compose framework automatically finds the peaks within a group of 1D XRD data files and runs a dash app for more precise peak detection.

The project contains standalone version of the [SplashML](./docs/concepts.md#SplashML) tagging framework in order to get a running demo.

To do this create the following directories:

* Clone this directory into any location
* Clone splash-ml at https://github.com/als-computing/splash-ml into the same directory 1D XRD exists in

Any data for automatic tagging should be under the data file.

## Running
First, let's install docker:

* https://docs.docker.com/engine/install/
* Next go into the 1D XRD dir
* type `docker-compose up` into your terminal (on more recent [versions](https://docs.docker.com/compose/#compose-v2-and-the-new-docker-compose-command) of the Docker desktop, `compose` is part of the docker CLI, meaning that the command is now `docker compose up`)

Next, open up the dash app or splash-ml API:

* Dash app: http://0.0.0.0:8050/
* Splash API: http://0.0.0.0:8000/api/splash_ml/docs

Notes:
- It is assumed that splash-ml handles the path to the dataset, such that file can be read, e.g. "C:/Users/..." vs "/mnt/c/Users/..."

See the [Tasks](./docs/tasks.md) for instructions on using the application.

# Copyright
MLExchange Copyright (c) 2021, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
