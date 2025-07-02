# Infrared Spectroscopy & Machine Learning Classification of Red-Stamp Inks

## Overview
A forensic workflow to classify commercial red-stamp inks on Hansol Copy™ paper using ATR-FTIR spectroscopy and machine learning. Five classifiers are compared—PLS-DA, k-NN, SVM, RF, and a feed-forward neural network (FNN)—across multiple spectral ranges and preprocessing methods. Model robustness is validated, and unknown samples are predicted.

## Data
All spectra are stored in **Spectra.xlsx**, containing four worksheets:

- **Entire** (4000–700 cm⁻¹)  
  Raw full-range spectra (1877 variables)  
- **Selected1** (1700–900 cm⁻¹)  
  Savitzky–Golay 2nd-derivative spectra (457 variables)  
- **Selected2** (1650–1100 cm⁻¹)  
  VIP-selected sub-range (314 variables)  
- **Unknown** (1700–900 cm⁻¹)  
  Second-derivative spectra of three unlabelled ink samples  

## Key Components

1. **Preprocessing**  
   L2 normalization and Savitzky–Golay 2nd-derivative + L2  

2. **Modeling Pipeline**  
   Three-fold cross-validation grid search over learning rate, optimizer, and hidden-unit count, followed by a final FNN trained on the full dataset  

3. **Validation & Metrics**  
   Macro-F1 score, 30 random-seed repeats (mean ± SD), y-scrambling control, and one-vs-rest ROC/AUC curves  

4. **Feature Selection**  
   VIP analysis identifies 1650–1100 cm⁻¹ as the most informative region  

5. **Unknown-Sample Prediction**  
   Final FNN applied to the “Unknown” sheet, exporting predicted softmax probabilities  

## File Structure

- **fnn_performance_comparison.R** &mdash; Main R analysis script  
- **Spectra.xlsx** &mdash; Spectral data (Entire, Selected1, Selected2, Unknown)  
- **Unknown_predictions.xlsx** &mdash; Output file with predicted probabilities  
- **README.md** &mdash; This documentation  

## Usage
To publish this project on GitHub, simply create a new repository on GitHub.com and use the “Upload files” button to add the contents of this folder (the R script, the Excel file, and this README). Once uploaded, commit the changes. 

If you prefer a local Git workflow, install Git on your computer, initialize this folder as a Git repository, add and commit all files, then connect it to your GitHub repository and push.

After uploading, other users can clone or download the repository, install the required R packages, and run the `fnn_performance_comparison.R` script to reproduce the analysis and generate predictions for the unknown inks.

## Key Findings

- **Best model**: FNN on 2nd-derivative spectra (1700–900 cm⁻¹) achieved perfect Test F1 (1.000) and AUC (1.000)  
- **Interpretable alternatives**: PLS-DA and RF also performed strongly (F1 ≥ 0.933)  
- **Critical sub-range**: 1650–1100 cm⁻¹  
- **Unknown inks**: High-confidence manufacturer predictions  

## License
This project is licensed under the MIT License.  
