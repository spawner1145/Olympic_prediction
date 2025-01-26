import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(filepath):
    """
    Loads, preprocesses, and splits the Olympic medal data.

    Args:
        filepath: Path to the 'summerOly_medal_counts.xlsx' file.

    Returns:
        X_train, X_test, y_train, y_test: Training and testing data.
    """

    # Load the data
    df = pd.read_excel(filepath)

    # Feature Engineering
    sport_cols = [col for col in df.columns if df[col].dtype != 'object' and col not in ['Rank', 'Gold', 'Silver', 'Bronze', 'Total', 'Year']]
    df['Total Events'] = df[sport_cols].sum(axis=1)
    df['Total Disciplines'] = df['Discipline'].apply(lambda x: 1 if x != 0 else 0)
    df['Total Sports'] = df[sport_cols].apply(lambda x: sum(x != 0), axis=1)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['NOC', 'Discipline'], drop_first=True)

    # Split into features (X) and target (y)
    X = df.drop(['Rank', 'Gold', 'Silver', 'Bronze', 'Total'], axis=1)
    y = df['Total']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test