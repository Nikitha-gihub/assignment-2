# Load Type Prediction ML Project

## Files
- `Load_Type_Prediction.ipynb` - complete Jupyter Notebook
- `load_type_prediction.py` - standalone runnable Python script
- `load_data(1).csv` - provided dataset
- `requirements.txt` - required Python packages

## Run in VS Code / Jupyter
1. Open the project folder.
2. Install packages:
   `pip install -r requirements.txt`
3. Open `Load_Type_Prediction.ipynb` and run all cells.

Or run:
`python load_type_prediction.py`

## Important validation rule
The last calendar month in the dataset is used as the final test set. The dataset supplied with this project ends in December 2018, so December 2018 is the test month after duplicate timestamps are removed.

## Expected result
On the supplied CSV, the Random Forest pipeline gives approximately:
- Accuracy: 0.9603
- Weighted precision: 0.9604
- Weighted recall: 0.9603
- Weighted F1: 0.9603

Exact results can vary if the dataset is changed.
