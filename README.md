# etMINFLUX-analysis
Analysis code for event-triggered MINFLUX data, including brief example data. Supporting the method implementation and findings in the preprint/paper "Event-triggered MINFLUX microscopy: smart microscopy to catch and follow rare events", Jonatan Alvelid, Agnes Koerfer, Christian Eggeling, bioRxiv (2025). ADD DOI HERE

An example minimal dataset for testing the analysis scripts is provided in this repository, while the full dataset supporting the findings of the manuscript (on which the same set of analysis scripts can be applied) can be found here: doi.org/10.5281/zenodo.15608840

The repository consists of Jupyter notebooks and python modules.

## Contents
exampledata - Folders with example data for the three experiment types, as well as confocal shift extraction data, used for testing the analysis scripts shared in this repository.

utility_functions - Convenience scripts for translating data or getting acquisition metadata from Imspector .msr files. 

scan_shift - Jupyter notebooks for fitting coordinate shifts to use for compensate confocal-MINFLUX data shifts.

metadata - Jupyter notebooks for getting metadata from experimenst - pipeline runtmes, pipeline parameter values, MINFLUX localizations metadata distributions.

cav1_analysis - Jupyter notebooks for running analysis and plotting analysis results, in ways presented in the connected manuscript, for cav1 events example data. With folder changes it can be applied to the full dataset separately provided in Zenodo. 

dyn1_analysis - Jupyter notebooks for running analysis and plotting analysis results, in ways presented in the connected manuscript, for dyn1 events example data. With folder changes it can be applied to the full dataset separately provided in Zenodo. 

gag_analysis - Jupyter notebooks for running analysis and plotting analysis results, in ways presented in the connected manuscript, for gag events example data. With folder changes it can be applied to the full dataset separately provided in Zenodo. 
