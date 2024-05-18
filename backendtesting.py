# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from utils import status_calc

def backtest():
    """
    A simple backtest that splits the dataset into a training set and a test set,
    then fits a Random Forest classifier to the training set. It prints the precision and accuracy
    of the classifier on the test set and compares the strategy's performance to passive investment in the S&P500.
    Note: There is a methodological flaw in this backtest that will give deceptively good results.
    These results should not encourage live trading.
    """
    # Load the dataset and drop rows with missing values
    data_df = pd.read_csv("keystats.csv", index_col="Date").dropna()

    # Define features and labels
    features = data_df.columns[6:]
    X = data_df[features].values
    y = status_calc(data_df["stock_p_change"], data_df["SP500_p_change"], outperformance=10)

    # Track returns
    z = data_df[["stock_p_change", "SP500_p_change"]].values

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size=0.2, random_state=0)

    # Instantiate and train the RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = clf.predict(X_test)
    print("Classifier Performance\n", "=" * 20)
    print(f"Accuracy score: {clf.score(X_test, y_test):.2f}")
    print(f"Precision score: {precision_score(y_test, y_pred):.2f}")

    # Calculate returns for predictions
    if np.any(y_pred):
        stock_returns = 1 + z_test[y_pred == 1, 0] / 100
        market_returns = 1 + z_test[y_pred == 1, 1] / 100

        avg_predicted_stock_growth = stock_returns.mean()
        avg_market_growth = market_returns.mean()
        percentage_stock_returns = (avg_predicted_stock_growth - 1) * 100
        percentage_market_returns = (avg_market_growth - 1) * 100
        total_outperformance = percentage_stock_returns - percentage_market_returns

        print("\nStock Prediction Performance Report\n", "=" * 40)
        print(f"Total Trades: {y_pred.sum()}")
        print(f"Average return for stock predictions: {percentage_stock_returns:.1f}%")
        print(f"Average market return in the same period: {percentage_market_returns:.1f}%")
        print(f"Our strategy outperforms the market by {total_outperformance:.1f} percentage points")
    else:
        print("No stocks predicted!")

if __name__ == "__main__":
    backtest()
