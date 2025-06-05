import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generate_dataset import generate_dataset, save_dataset
from src.train_model import train_and_evaluate


def setup_module(module):
    # Ensure dataset exists for tests
    save_dataset()


def test_dataset_exists():
    assert os.path.exists('data/sample_transactions.csv'), 'Dataset file missing'


def test_model_training():
    df = generate_dataset()
    acc = train_and_evaluate(df)
    assert 0 <= acc <= 1
