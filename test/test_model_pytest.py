from pathlib import Path
from src.model import train_and_save


def test_train_and_save_model(tmp_path: Path):
    csv_path = tmp_path / "iris.csv"
    model_path = tmp_path / "rf.joblib"

    result = train_and_save(
        csv_path=csv_path,
        model_path=model_path,
        test_size=0.2,
        random_state=42
    )

    assert result.model_path.exists(), "Model artifact was not saved"
    assert result.accuracy >= 0.85, f"Accuracy too low: {result.accuracy}"
