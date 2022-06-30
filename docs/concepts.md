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
This option in the application uses [Bayesian block inference approach](https://iopscience.iop.org/article/10.1088/0004-637X/764/2/167/meta) that splits the input data in blocks, where each block contains 1 peak. Hence, this approach auto-detect the number of peaks in the data and their approximate locations within the block boundaries in the x-axis. This option is available in the application and is also appropriate for the batch processing of files.

## Peak Location and Curve Fitting
Given the number of peaks, and boundaries conditions in the Blocks approach, the location of the peaks is estimated by using the continuos wavelet transform (CWT) of the input data to detect its relative maximum points (peaks). To estimate the Full Width Half Maximum (FWHM), we perform curve fitting by first estimating an individual curve profile per peak, where their initial shape parameters are estimated through the CWT of the input data. The sum of these curve profiles is represented as the `unfit` curve in the plot. We then fit the `unfit` curve through a Simplex algorithm with a least squares function to obtain the `fit` curve in the plot. In addition, we plot the `residual` curve, which represents the absolute value of the difference between the `fit` and input curve (`XRD Data`). If the application of a baseline was selected, this `baseline` will be plotted in the graph too.

## Error Flags
If the mean ratio of the `residual` and input curves is greater than 10%, the peaks are flagged by adding a `(F)` right next to the tag name.
 
