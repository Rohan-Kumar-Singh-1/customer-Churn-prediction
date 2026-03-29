import numpy as np
from src.model import build_model


def test_model_output_shape():
    model = build_model(10)
    dummy = np.random.rand(5, 10)

    output = model.predict(dummy)

    assert output.shape == (5, 1)