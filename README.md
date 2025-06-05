# E-Commerce Consumer Behaviour Prediction

This project demonstrates a simple workflow for generating a synthetic e-commerce
transactions dataset and training a logistic regression model to predict whether
a customer will make a purchase.

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- pytest (for running tests)

Install dependencies with:

```bash
pip install pandas numpy scikit-learn pytest
```

## Usage

1. **Generate the dataset**
   ```bash
   python src/generate_dataset.py
   ```
   This will create `data/sample_transactions.csv`.

2. **Train the model**
   ```bash
   python src/train_model.py
   ```
   The script will print the accuracy of the trained model.

3. **Run tests**
   ```bash
   pytest -q
   ```

The dataset is randomly generated, so results may vary between runs.
