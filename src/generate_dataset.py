"""Generate a synthetic e-commerce transactions dataset."""

import numpy as np
import pandas as pd


def generate_dataset(n: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 70, size=n),
        'Gender': np.random.randint(0, 2, size=n),  # 0=Female, 1=Male
        'BrowsingTime': np.random.normal(10, 5, size=n).clip(min=0),
        'PagesVisited': np.random.poisson(5, size=n),
        'AddToCart': np.random.randint(0, 2, size=n)
    }

    prob_purchase = (
        0.3 * (data['Age'] > 30).astype(int)
        + 0.2 * data['Gender']
        + 0.2 * (data['BrowsingTime'] > 10).astype(int)
        + 0.3 * data['AddToCart']
    )
    labels = (prob_purchase + np.random.rand(n) > 1.2).astype(int)

    df = pd.DataFrame(data)
    df['Purchase'] = labels
    return df


def save_dataset(path: str = 'data/sample_transactions.csv', n: int = 1000) -> None:
    df = generate_dataset(n)
    df.to_csv(path, index=False)
    print(f'Dataset saved to {path}')


if __name__ == '__main__':
    save_dataset()
