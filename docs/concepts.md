# Concepts

## Apply Baseline to Peak Fitting
An option in the user interface to change the lower baseline of the fitting.

## Peak Shape
This option in the application allows the user to determine the curve shape for fitting peaks. Currently supported are `Guassian` and `Voigt`.

## SplashML
[SplashML](https://github.com/als-computing/splash-ml) is database service for storing tag information about datasets. It does not store datasets themselves, rather links to those datasets and their corresponding tags. This project contains a standalone version of SplashML for demonstration purposes.

## Tag with Window
This option in the application uses an algorithm that allows the user to preselect the number of peaks to detect.
## Tag with Blocks
This option in the application uses [Bayesian block inference approach](https://iopscience.iop.org/article/10.1088/0004-637X/764/2/167/meta) that auto-detects the number of peaks. This option is available in the application and is also appropriate for the batch processing of files.
