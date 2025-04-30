import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate



#======================= Feature Selection functions ========================#
def feature_selection(X, y, n_estimators=100, random_state=42):
    """
    Perform feature selection using Random Forest based on OOB feature importance.
    
    Parameters:
    - X (DataFrame): Feature dataset
    - y (Series): Output variable
    - n_estimators (int): Number of trees in the forest (default=100)
    - random_state (int): Random state for reproducibility (default=42)
    
    Returns:
    - removed_features (list): List of removed features in the order they were dropped
    - rmse_list (list): RMSE values for each iteration
    - r2_list (list): R2 values for each iteration
    """

    X_loop = X.copy()  # avoid modifying the original data
    
    # Initialize lists to store results
    rmse_list = []
    r2_list = []
    important_features = []
    
    # Set up the Random Forest model with OOB enabled
    model = RandomForestRegressor(n_estimators=n_estimators, 
                                  oob_score=True, 
                                  random_state=random_state)
    
    # Loop until there are no features left
    while X_loop.shape[1] > 0:
        # Fit the model on the current dataset
        model.fit(X_loop, y)
        
        # Calculate OOB RMSE and R2
        oob_rmse = np.sqrt(mean_squared_error(y, model.oob_prediction_))
        oob_r2 = r2_score(y, model.oob_prediction_)
        
        # Record the performance metrics
        rmse_list.append(oob_rmse)
        r2_list.append(oob_r2)
        
        # Determine feature importances and identify the least important feature
        feature_importances = model.feature_importances_
        least_important_feature = X_loop.columns[np.argmin(feature_importances)]
        important_features.append(least_important_feature)
    
        # Drop the least important feature
        X_loop = X_loop.drop(columns=[least_important_feature])
    
    print("Feature Selection Complete")
    
    # Return the results as a dictionary
    return {
        "important_features": important_features,
        "rmse_list": rmse_list,
        "r2_list": r2_list
    }




# Mapping of weather codes to descriptive names
synop_mapping = {
    0: 'Clear',
    3: 'Dust Storm',
    4: 'Fog',
    5: 'Drizzle',
    6: 'Rain',
    7: 'Snow',
    8: 'Showers'
}


def plot_feature_selection_base(important_features, rmse_list, r2_list, file_prefix, synop_mapping=synop_mapping, point=None):
    """
    Base function to plot RMSE and R2 scores by the remaining important features.

    Parameters:
    - important_features (list): List of retained important features in order of selection.
    - rmse_list (list): RMSE values for each iteration of feature selection.
    - r2_list (list): R2 values for each iteration of feature selection.
    - file_prefix (str): Prefix for the filename to save the plot.
    - synop_mapping (dict): Mapping of weather codes to descriptive names.
    - point (str, optional): Feature name where the tuning point is applied (for green line).
    """
    # Extract model type, channel, and weather code from file_prefix
    parts = file_prefix.split('_')
    channel = parts[0]  # "FSO" or "RF"
    model_type = parts[1]  # "Generic" or "Specific"/ "Method2" or "Method3"
    weather_code = int(parts[2]) if "Specific" in model_type else None
    weather_name = synop_mapping.get(weather_code, "Unknown") if weather_code is not None else "All Weather"

    # Set a style for white background with black grid
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a figure and primary y-axis
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot RMSE on the primary y-axis
    ax1.plot(important_features, rmse_list, marker='o', color='b', label='OOB RMSE', linewidth=2, markersize=8)
    ax1.set_ylabel('OOB RMSE (dB)', color='b', fontsize=24)
    ax1.tick_params(axis='y', labelcolor='b', labelsize=24)
    ax1.set_xticks(range(len(important_features)))
    ax1.set_xticklabels(important_features, rotation=90, fontsize=20)
    ax1.grid(visible=True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Add a vertical green line if a tuning point is provided
    if point:
        x_index = important_features.index(point) + 0.5
        ax1.axvline(x=x_index, color='green', linestyle='--', linewidth=4, label='Threshold')

    # Create a secondary y-axis for R2
    ax2 = ax1.twinx()
    ax2.plot(important_features, r2_list, marker='o', color='r', label='OOB R2', linewidth=2, markersize=8)
    ax2.set_ylabel('OOB R2', color='r', fontsize=22)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=22)
    ax2.grid(visible=False)

    # Add title
    plt.title(f"{channel} Channel - {model_type} Model - {weather_name}", fontsize=22)
    fig.tight_layout()

    return fig, ax1


def plot_feature_rank(important_features, rmse_list, r2_list, file_prefix, synop_mapping=synop_mapping):
    """
    Plot RMSE and R2 scores by the remaining important features without a green line.
    """
    fig, ax1 = plot_feature_selection_base(important_features, rmse_list, r2_list, file_prefix, synop_mapping)
    
    # Save the plot
    output_dir = 'Figure/Feature_Selection'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_feature_selection_results.eps'), format='eps')
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_feature_selection_results.jpg'), format='jpg')
    plt.show()


def plot_green_line(important_features, rmse_list, r2_list, file_prefix, point, synop_mapping=synop_mapping):
    """
    Plot RMSE and R2 scores by the remaining important features with a green line.
    """
    fig, ax1 = plot_feature_selection_base(important_features, rmse_list, r2_list, file_prefix, synop_mapping, point=point)
    
    # Save the plot
    output_dir = 'Figure/Feature_Selection'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_greenline_results.eps'), format='eps')
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_greenline_results.jpg'), format='jpg')
    plt.show()


def save_important_features(important_features, point, file_prefix, output_dict):
    """
    Save the important features for each model after the turning point in the desired format.

    Parameters:
    - important_features (list): List of retained important features in order of selection.
    - point (str): Feature name where the turning point is applied.
    - file_prefix (str): Prefix for the filename to identify the model (e.g., "FSO_Specific0").
    - output_dict (dict): Dictionary to store the important features for each model.
    """
   
    # Extract model type, channel, and weather code from file_prefix
    parts = file_prefix.split('_')
    channel = parts[0]  # "FSO" or "RF"
    model_type = parts[1]  # "Generic" or "Specific"/ "Method2" or "Method3"


    # Check if the turning point exists in the important features
    if point in important_features:
        # Slice the features up to and including the turning point
        updated_features = important_features[important_features.index(point) + 1:]

        # Save the updated features into the output dictionary
        if channel not in output_dict:
            output_dict[channel] = {}
        output_dict[channel][model_type] = updated_features

        # Print the updated features for the model
        print(f"Updated important features for {channel} - {model_type}: {updated_features}")
    else:
        print(f"Warning: Turning point '{point}' not found in important features for {file_prefix}.")




#======================= Hyperparameter tuning functions ========================#
def tune_random_forest(X_train, y_train, cv_folds=5, random_state=42, n_jobs=-1):
    """
    Tune hyperparameters for a Random Forest Regressor using GridSearchCV and evaluate the best model.
    
    Parameters:
    - X_train (DataFrame): Training feature set
    - y_train (Series): Training target variable
    - cv_folds (int): Number of cross-validation folds (default is 5)
    
    Returns:
    - best_params_rf (dict): Best hyperparameters for the Random Forest model
    - rf_mean_rmse (float): Mean RMSE for the best model after cross-validation
    - rf_std_rmse (float): Standard deviation of RMSE for the best model after cross-validation
    """

    # Define the parameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [50,100,200],    # Number of trees in the forest
        'max_depth': [10, 20, None],       # Maximum depth of each tree
        'min_samples_split': [2, 5, 10],   # Minimum samples required to split a node
        'min_samples_leaf': [1, 2, 4]     # Minimum samples required at a leaf node
    }
    
      
    # Initialize Random Forest Regressor
    rf = RandomForestRegressor(random_state=random_state)
    
    # Perform grid search
    rf_grid_search = GridSearchCV(
        estimator=rf,
        param_grid=rf_param_grid, 
        cv=cv_folds, 
        scoring='neg_root_mean_squared_error', 
        return_train_score=True, 
        n_jobs=n_jobs
        )
    rf_grid_search.fit(X_train, y_train)
    
    # Get the best parameters and best score
    best_params_rf = rf_grid_search.best_params_
    best_score_rf = -rf_grid_search.best_score_
    
    # Perform cross-validation for the best Random Forest model
    rf_best_model = rf_grid_search.best_estimator_
    cv_results_rf = cross_validate(
        estimator=rf_best_model, 
        X=X_train, 
        y=y_train, 
        cv=cv_folds, 
        scoring='neg_root_mean_squared_error', return_train_score=True,
        n_jobs=n_jobs
        )
    
    rf_mean_rmse = -np.mean(cv_results_rf['test_score'])
    rf_std_rmse = np.std(cv_results_rf['test_score'])

    # Print results for clarity
    print(f"Best Parameters: {best_params_rf}")
    print(f"Mean RMSE: {rf_mean_rmse:.4f}")
    print(f"Standard Deviation of RMSE: {rf_std_rmse:.4f}")
    
    return best_params_rf, rf_mean_rmse, rf_std_rmse