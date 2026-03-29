import pandas as pd
from src.preprocessing import preprocess_data


def test_missing_values():
    df = pd.DataFrame({
        "A": [1, None, 3],
        "B": ["x", "y", None],
        "Churn": [0, 1, 0]
    })

    X, y = preprocess_data(df, "Churn")

    assert X.shape[0] == 3