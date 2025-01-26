import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_rf(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates a Random Forest model.

    Args:
        X_train, X_test, y_train, y_test: Training and testing data.

    Returns:
        rf_model: The trained Random Forest model.
    """

    # Initialize and train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Random Forest Model Evaluation:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2 Score: {r2:.2f}")

    # Feature Importance
    importances = rf_model.feature_importances_

    # Load the original dataset to get the original feature names
    original_df = pd.read_excel('summerOly_medal_counts.xlsx')
    original_feature_names = ['Year'] + [col for col in original_df.columns if col not in ['Rank', 'Gold', 'Silver', 'Bronze', 'Total', 'Year', 'NOC', 'Discipline']] + ['Total Events', 'Total Disciplines', 'Total Sports']

    # Get one-hot encoded feature names from the DataFrame used for training
    encoded_feature_names = []
    for col in original_df.columns:
        if col == 'NOC' or col == 'Discipline':
            # Get unique values from the training data
            unique_values = set()
            for _, row in original_df.iterrows():
                if row['Year'] in y_train.index:
                    unique_values.add(row[col])

            for value in sorted(unique_values):  # Sort for consistent order
                encoded_feature_names.append(f"{col}_{value}")
        else:
            encoded_feature_names.append(col)

    encoded_feature_names = ['Year'] + [col for col in encoded_feature_names if col not in ['Rank', 'Gold', 'Silver', 'Bronze', 'Total', 'Year', 'NOC', 'Discipline'] and col in original_df.columns] + ['Total Events', 'Total Disciplines', 'Total Sports']

    # Create mapping from feature name to importance
    feature_importance_map = {name: importance for name, importance in zip(encoded_feature_names, importances)}

    # Aggregate feature importances for NOC and Discipline
    noc_importance = sum(feature_importance_map.get(name, 0) for name in encoded_feature_names if name.startswith('NOC_'))
    discipline_importance = sum(feature_importance_map.get(name, 0) for name in encoded_feature_names if name.startswith('Discipline_'))

    # Filter out the individual one-hot encoded features for NOC and Discipline
    aggregated_importances = [
        importance for name, importance in feature_importance_map.items()
        if not name.startswith('NOC_') and not name.startswith('Discipline_')
    ]
    aggregated_feature_names = [
        name for name in encoded_feature_names
        if not name.startswith('NOC_') and not name.startswith('Discipline_')
    ]

    # Add the aggregated importances for NOC and Discipline
    aggregated_importances.extend([noc_importance, discipline_importance])
    aggregated_feature_names.extend(['NOC', 'Discipline'])

    # Sort the features by importance
    sorted_idx = np.argsort(aggregated_importances)[::-1]
    sorted_importances = np.array(aggregated_importances)[sorted_idx]
    sorted_feature_names = np.array(aggregated_feature_names)[sorted_idx]

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_feature_names)
    plt.title("Feature Importances in Random Forest Model")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    return rf_model