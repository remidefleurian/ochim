[![license](https://img.shields.io/github/license/remidefleurian/ochim)](https://github.com/remidefleurian/ochim/blob/master/LICENSE)

# [Analysis] Onsets of Chills in Music (oChiM)

This repository reproduces the analysis reported in the following article:

> Melodic expectation as an elicitor of music-evoked chills  
> Rémi de Fleurian, Ana Clemente, Emmanouil Benetos, Marcus T. Pearce  
> Nature Communications

The analysis takes the [oChiM dataset](https://doi.org/10.17605/osf.io/x59fm) provided with the article as input, and returns the evaluation metrics for the best SVM classifier as output. It consists of two main scripts:

- `prep.R` reshapes the raw data, runs the PCA, assigns k-folds, augments the onsets of chills, and exports the resulting pre-processed data
- `svm.py` takes the pre-processed data (already provided in this repository or overwritten by the R script), runs the SVM classifier, and computes model evaluation metrics

Detailed steps are provided below to set up the required R and Python environments and run the scripts. 

---

## Reproducibility note

Model performance metrics may vary slightly depending on your local R environment and CPU architecture. For instance, execution on Apple Silicon hardware resulted in marginal shifts in recall (+0.013) and balanced accuracy (+0.002) compared to the older Intel hardware used for the metrics reported in the article. This is most likely attributable to architecture-specific handling of floating-point arithmetic affecting the exact values returned by the principal component analysis.

---

## Initial setup

```bash
# clone repository
git clone https://github.com/remidefleurian/ochim.git

# navigate to directory
cd ochim
```

---

## R scripts (optional)

The R scripts can be skipped, in which case the Python script will use the pre-processed data provided in this repository.

### Download raw data

- Download [features.zip](https://osf.io/x59fm/files/k6bmv) and extract it in the data folder
- Download [ochim.csv](https://osf.io/x59fm/files/46mkj) and place it in the data folder

The folder should now look like this:
```
data/
├── features/
│	├── segmented-200-ms/
│	└── segmented-500-ms/
├── preprocessed/
│	├── k1.csv
│	├── k2.csv
│	├── k3.csv
│	├── k4.csv
│	└── k5.csv
└── ochim.csv
```

### Run R scripts

You can probably skip the environment setup if you already have R version ≥ 4.1.0 and tidyverse version ≥ 2.0.0.

```bash
# environment setup
Rscript envs/chills.R

# data pre-processing script
Rscript prep.R
```

--- 

## Python script

### Environment setup

Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) to proceed. Instructions are only provided for Mac, but should be available online for Windows and Linux (dependencies are listed in `envs/chills.yml`).

```bash
# with an intel mac
conda env create -f envs/chills.yml
conda activate chills
```

```bash
# with an apple silicon mac
CONDA_SUBDIR=osx-64 conda create -n chills
conda activate chills
conda config --env --set subdir osx-64
conda env update -f envs/chills.yml
```

### Run Python script

The script will take a moment to start if the environment was set up for an Apple Silicon Mac. It will then run for a while, with progress being printed in the terminal. Results will be stored in `output/results/results.csv`.

```bash
# svm script
python svm.py
```
