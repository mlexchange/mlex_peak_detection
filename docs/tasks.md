# How To Guide

The peak detection appplication is a browser-based framework for automatically detecting peaks in 1-D XRD data set. The user can choose from several different algorithms to guide detection, which differ in the amount of user intervention vs. automation that can be performed. 

## Data format
Currently, the 1D XRD demo accepts a 2 column comma-separated (.csv) files, with no header row, for example

```csv
1.502500072169232093e+00,1.333536761775019386e+01
1.507500072883772191e+00,1.558678643552360654e+01
1.512500073598311845e+00,1.727255455621030933e+01
1.517500074312851721e+00,1.817530381724509425e+01
```
## Single File Detection
The 1D XRD application lets you provide a file, label detect peaks, and store the those peaks as features in the [SplashML](./concepts.md#SplashML) database.

Steps:
* Click on the "Select Files" link

* Browse to a file on your file system.

> The application displays a plot of the file, with two panes, the full plot and plot with selectors to zoom into a particular section

* Optional: select [Apply Baseline to Peak Fitting](./concepts.md#Apply_Baseline_to_Peak_Fitting)

* Choose a peak detection method:

  * Option 1: [Window](./concepts.md#Tag_with_Window). This option requires you to also add a value in "Number of Peaks".

  * Option 2: [Blocks](./concepts.md#Tag_with_Blocks).

* Enter a tag name. This name will be used to generate the names of the tags added to [SplashML](./concepts.md#SplashML).

* Choose the [Peak Shape](./concepts.md#Peak_Shape). 

### What happened?
The application displays the detected peaks in the graph and `Current Tags` table. Each tag in the `Current Tags` can be saved into SplashML as a new tag, with the `Peak` location (midpoint and amplitude) and the estimated Full Width Half Max of the peak. A color was assigned to each tag, which matches the color in the graph. 

Additionally, the plot displays the [unfit](./concepts.md#Peak_Detection_and_Curve_Fitting), [fit](./concepts.md#Peak_Detection_and_Curve_Fitting), [residual](./conceps.md#Peak_Detection_and_Curve_Fitting), and, if selected, [baseline](./concepts.md#Peak_Detection_and_Curve_Fitting) curves.

One can now `Download Tags`, which downloads the table of detected peaks as a csv file or `Save Tags to SplashML`, which inserts them into SplashML.

## Batch Detection
A future feature which allows the user to bulk detect peaks on a number of files all at once.
