# Infrared Spectroscopy & Machine Learning Classification of Red Stamp Inks

## Overview
This project implements a forensic workflow to classify red stamp (seal) inks on questioned documents using ATR-FTIR spectroscopy and machine learning. By combining spectral preprocessing, dimensionality reduction, and supervised classification models, we aim to provide a rapid, nondestructive, and objective tool for forensic ink analysis.

## Data
The dataset consists of ATR-FTIR spectra collected from individual stamp impressions on Hansol Copy™ paper. Ten commercial red stamp ink products from five countries were acquired, with 8–32 distinct product lots per brand (128 total spectra). Three additional “unknown” ink samples were included for final model predictions. Raw and Savitzky–Golay second-derivative spectra are stored in `Summary.xlsx` (sheets: “Summary” & “Selected”).

## Key Features
- **Data Import & L2 Normalization**: Load spectral data from Excel and apply Euclidean norm normalization.  
- **Savitzky–Golay 2nd-Derivative Filtering**: Enhance spectral resolution and reduce baseline effects.  
- **Principal Component Analysis (PCA)**: Visualize high-dimensional spectra in 2D space.  
- **Supervised Classification**:
  - Partial Least Squares Discriminant Analysis (PLS-DA)
  - k-nearest neighbors algorithm (k-NN)
  - Support Vector Machine (SVM) 
  - Random Forest (RF)    
  - Feed-Forward Neural Network (FNN)  
- **Feature Importance Analysis**: Identify critical wavenumber ranges (1650–1200 cm⁻¹) driving model decisions.  
- **Validation**: Stratified train/test split, three-fold cross-validation, and F1-score metrics.  
- **Unknown Sample Prediction**: Classify three unlabelled inks using the final FNN model.

## Paper Information
This code supports the manuscript:  
**“Infrared Spectroscopy and Machine Learning for Classification of Red Stamp Inks on Questioned Documents”**  
Submitted to *Journal of Chemometrics*.

## Paper Link
The preprint and supplementary information will be made available upon publication.

