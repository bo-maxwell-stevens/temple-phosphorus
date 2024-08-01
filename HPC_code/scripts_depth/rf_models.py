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
parser.add_argument('--soil_test', required=True, choices=['H3A', 'M3', 'Ols'], help='Soil test to use.')
parser.add_argument('--depth', required=True, choices=['0_2', '2_6'], help='Depth to use.')
parser.add_argument('--analysis_type', required=True, choices=['topsoil', 'bottomsoil', 'all_topsoil', 'all_bottomsoil'], help='Analysis type to use.')
parser.add_argument('--phosphorus_treatment', type=int, required=True, help='Whether to use phosphorus treatment.')
args = parser.parse_args()

# Convert args.phosphorus_treatment to boolean
args.phosphorus_treatment = bool(args.phosphorus_treatment)

data = pd.read_csv('../Data/combined_zones.csv')

# Columns that will always be included
base_cols = []
if args.phosphorus_treatment:
    base_cols.append('PhosphorusTreatment')

# Filter data for current iteration
temp_data = data[(data['FieldID'] == args.field) & (data['Year'] == args.year)]

# Select columns for this iteration
if args.analysis_type == 'topsoil':
    feature_cols = [f'{args.soil_test}_P_{args.depth}'] + base_cols
elif args.analysis_type == 'bottomsoil':
    feature_cols = [f'{args.soil_test}_P_{args.depth}'] + base_cols
elif args.analysis_type == 'all_topsoil':
    feature_cols = [col for col in temp_data.columns if col.startswith(args.soil_test) and col.endswith('0_2')] + base_cols
elif args.analysis_type == 'all_bottomsoil':
    feature_cols = [col for col in temp_data.columns if col.startswith(args.soil_test) and col.endswith('2_6')] + base_cols

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

# Save model and results
model_path = f"../rf_model_output_depth/models/{args.field}_{args.year}_{args.soil_test}_{args.depth}_{args.analysis_type}_{'p' if args.phosphorus_treatment else 'np'}.joblib"
dump(best_model, model_path)

# Save results
results = {
    'model_name': model_path,
    'num_input_params': len(feature_cols),
    'r2': r2,
    'mse': mse,
    'features_importances': {feature: importance for feature, importance in zip(feature_cols, best_model.feature_importances_)},
}

results_path = f"../rf_model_output_depth/results/{args.field}_{args.year}_{args.soil_test}_{args.depth}_{args.analysis_type}_{'p' if args.phosphorus_treatment else 'np'}.txt"
x_train_path = f"../rf_model_output_depth/X/{args.field}_{args.year}_{args.soil_test}_{args.depth}_{args.analysis_type}_{'p' if args.phosphorus_treatment else 'np'}.csv"

pd.DataFrame(X_train).to_csv(x_train_path, index=False)

with open(results_path, 'wt') as f:
    json.dump(results, f)