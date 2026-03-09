from dataclasses import dataclass

@dataclass
class Config:
    asset: str = "BTC-USD"
    start_date: str = "2021-01-01"
    interval: str = "1d"
    test_size: float = 0.2
    target_horizon: int = 1
    random_state: int = 42

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    outputs_dir: str = "outputs"