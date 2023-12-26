import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import dump
import json
import argparse

# Define your parameter grid here

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

parser = argparse.ArgumentParser(description='RF model with specific parameters.')
parser.add_argument('--field', required=True, help='Field to use.')
parser.add_argument('--year', type=int, required=True, help='Year to use.')
parser.add_argument('--zone', required=True, choices=['Zone_1', 'Zone_2', 'Zone_3', 'Zone_4'], help='Soil test to use.')
parser.add_argument('--phosphorus_treatment', type=int, required=True, help='Whether to use phosphorus treatment.')
args = parser.parse_args()

# convert args.phosphorus_treatment to boolean
args.phosphorus_treatment = bool(args.phosphorus_treatment)

data = pd.read_csv('../Data/combined_zones.csv')

#### For control plots only, if doing entire dataset, uncomment the next line
# data = data[data['PhosphorusTreatment'] == 0]

# Columns that will always be included
if args.phosphorus_treatment == True:
    print('Yes')
    base_cols = [ 'TOC_0_2', 'IC_0_2', 'TN_0_2', 'TOC_2_6', 'IC_2_6', 'TN_2_6','PhosphorusTreatment'] #'elevation', 'slope', 'aspect', 'TC_0_2', 'TC_2_6', 
    #base_cols = ['PhosphorusTreatment']
else: 
    print('No')
    base_cols = ['TOC_0_2', 'IC_0_2', 'TN_0_2', 'TOC_2_6', 'IC_2_6', 'TN_2_6']#'elevation', 'slope', 'aspect', 'TC_0_2', 'TC_2_6', 
    #base_cols = []

# Filter data for current iteration
temp_data = data[(data['FieldID'] == args.field) & (data['Year'] == args.year)]

temp_data = temp_data[temp_data[args.zone] == 1]

feature_cols = [col for col in temp_data.columns if col.startswith('Ols')] + [col for col in temp_data.columns if col.startswith('H3A')] + [col for col in temp_data.columns if col.startswith('M3')] + base_cols

X = temp_data[feature_cols]
y = temp_data['Yield']

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit model
rf = RandomForestRegressor()
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Compute metrics on test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save model and results (change folder name depending on type of output -- controls versus full dataset)
if args.phosphorus_treatment == True:
    model_path = f"../rf_model_output_zones_spatial/models/{args.field}_{args.year}_{args.zone}_p.joblib"
else:
    model_path = f"../rf_model_output_zones_spatial/models/{args.field}_{args.year}_{args.zone}_np.joblib"

dump(best_model, model_path)

# Save results
results = {
    'model_name': model_path,
    'num_input_params': len(feature_cols),
    'r2': r2,
    'mse': mse,
    'features_importances': {feature: importance for feature, importance in zip(feature_cols, best_model.feature_importances_)},
}

if args.phosphorus_treatment == True:
    results_path = f"../rf_model_output_zones_spatial/results/{args.field}_{args.year}_{args.zone}_p.txt"
    x_train_path = f"../rf_model_output_zones_spatial/X/{args.field}_{args.year}_{args.zone}_p.csv"
else:
    results_path = f"../rf_model_output_zones_spatial/results/{args.field}_{args.year}_{args.zone}_np.txt"
    x_train_path = f"../rf_model_output_zones_spatial/X/{args.field}_{args.year}_{args.zone}_np.csv"

pd.DataFrame(X_train).to_csv(x_train_path, index = False)

print(results_path)
with open(results_path, 'wt') as f:
    json.dump(results, f)
