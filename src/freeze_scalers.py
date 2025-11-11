# freeze_scalers.py
from pathlib import Path
import joblib
from data_loader import get_data_and_scalers  # 你现有的
import config  # 含 LOG_TRANSFORMED_COLS 那个

SRC_DIR = Path(__file__).resolve().parent / "src"
RESULTS_DIR = SRC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

for opamp in ["5t_opamp", "two_stage_opamp"]:
    data = get_data_and_scalers(opamp_type=opamp)
    x_scaler = data["x_scaler"]
    y_scaler = data["y_scaler"]

    joblib.dump(x_scaler, RESULTS_DIR / f"{opamp}_x_scaler.gz")
    joblib.dump(y_scaler, RESULTS_DIR / f"{opamp}_y_scaler.gz")
    print(f"saved scalers for {opamp}")
