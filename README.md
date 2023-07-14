# MachineLearning_PDAC_Prediction
Risk Prediction of Pancreatic Cancer Using Urine Biomarkers
By: Jenny Harston

Tech specs:
Development environment: Anaconda3 - Python 3.9.7

Packages used: 
•	pandas
•	numpy
•	random
•	matplotlib
•	sklearn

Files: 
•	project_JennyHarston.py
•	pc_data.csv

Instructions:
•	Include both files in the same folder
•	Make sure all packages used are installed in your environment
•	Run the code using python
•	If you would like to look at the first 10 samples of the original dataset, uncomment line 113
•	The code is currently set up to compare the diagnoses of Control vs. PDAC. If you would like to compare Benign vs. PDAC, change line 119 to 
dataset = CompareDiagnoses(diag_dict, 2, 3)
 
•	The code is currently set up to display the ROC curve from an estimator. If you would like to display the ROC curve from the predictions, comment out lines 161-167, and uncomment lines 169-175
 
•	Model prediction evaluations will be displayed in the terminal and the ROC curve figure will be produced.

