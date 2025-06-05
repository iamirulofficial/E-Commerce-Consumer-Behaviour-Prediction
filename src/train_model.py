import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def load_data(path: str = 'data/sample_transactions.csv') -> pd.DataFrame:
    """Load the transaction dataset from CSV."""
    return pd.read_csv(path)


def build_pipeline() -> Pipeline:
    """Create the ML pipeline."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    return pipeline


def train_and_evaluate(df: pd.DataFrame) -> float:
    X = df.drop('Purchase', axis=1)
    y = df['Purchase']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_pipeline()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc


def main():
    df = load_data()
    acc = train_and_evaluate(df)
    print(f'Accuracy: {acc:.3f}')


if __name__ == '__main__':
    main()
