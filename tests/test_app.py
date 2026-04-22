import sys
from unittest.mock import MagicMock

# Mock dependencies that are not installed in the environment
mock_gradio = MagicMock()
mock_pandas = MagicMock()
mock_sklearn_ensemble = MagicMock()

sys.modules["gradio"] = mock_gradio
sys.modules["pandas"] = mock_pandas
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.ensemble"] = mock_sklearn_ensemble

import pytest
import app

def test_load_model_missing_file():
    # Setup mock to raise FileNotFoundError
    mock_pandas.read_csv.side_effect = FileNotFoundError()

    with pytest.raises(FileNotFoundError) as excinfo:
        app.load_model()

    assert "Dataset file 'star_classification.csv' not found" in str(excinfo.value)

def test_predict_logic():
    # Mock the model object
    mock_model = MagicMock()
    app.model = mock_model

    # Test STAR prediction
    mock_model.predict.return_value = ["STAR"]
    result = app.predict(1, 2, 3, 4, 5, 0.1)
    assert result == "⭐ STAR"

    # Test GALAXY prediction
    mock_model.predict.return_value = ["GALAXY"]
    result = app.predict(1, 2, 3, 4, 5, 0.1)
    assert result == "🌌 GALAXY"

    # Test QSO prediction
    mock_model.predict.return_value = ["QSO"]
    result = app.predict(1, 2, 3, 4, 5, 0.1)
    assert result == "💫 QSO"
