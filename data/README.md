# Dataset placement

Place the **PIMA Indians Diabetes** CSV here as `data/diabetes.csv`.
(Original source: National Institute of Diabetes and Digestive and Kidney Diseases; widely mirrored on Kaggle/UCI.)

Expected structure: 768 rows, header row, 9 columns:

```
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
...
```

`Outcome`: 0 = non-diabetic class, 1 = diabetic class.
The data is NOT bundled with this project and no data is generated or simulated for real runs.

## Getting the file
The reference notebook loads it from the Kaggle dataset folder `pima-indians-diabetes-database` (file `diabetes.csv`).
Download it from Kaggle (Kaggle CLI: `kaggle datasets download -d uciml/pima-indians-diabetes-database`), unzip, and copy
`diabetes.csv` here. The `.ipynb` notebook itself does not contain the data (only a 5-row preview).
