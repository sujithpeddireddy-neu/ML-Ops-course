from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


@dataclass
class TrainResult:
    accuracy: float
    model_path: Path


def load_or_create_iris(csv_path: str | Path) -> pd.DataFrame:
    """
    Loads iris from CSV if it exists; otherwise generates it from sklearn and writes it.
    """
    csv_path = Path(csv_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    d = load_iris(as_frame=True)
    df = d.frame
    df.to_csv(csv_path, index=False)
    return df


def train_and_save(
    csv_path: str | Path = "data/iris.csv",
    model_path: str | Path = "data/random_forest_iris.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    """
    Trains a RandomForest model on iris and saves it as a joblib artifact.
    Returns accuracy + saved model path.
    """
    df = load_or_create_iris(csv_path)

    # Iris frame includes target column named "target"
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=4,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return TrainResult(accuracy=float(acc), model_path=model_path)


if __name__ == "__main__":
    result = train_and_save()
    print(f"Saved model to: {result.model_path}")
    print(f"Accuracy: {result.accuracy:.4f}")
