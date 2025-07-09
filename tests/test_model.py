import pytest
from model.ann import load_model  # example function

def test_model_loading():
    model = load_model()
    assert model is not None
