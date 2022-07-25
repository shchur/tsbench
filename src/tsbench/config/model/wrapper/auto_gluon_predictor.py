import json
import os
import warnings
from distutils.dir_util import copy_tree
from pathlib import Path
from typing import Iterator, Optional
from gluonts.dataset.common import Dataset
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import Predictor

try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
except ImportError:
    TimeSeriesPredictor = None

AUTOGLUON_IS_INSTALLED = TimeSeriesPredictor is not None

USAGE_MESSAGE = """
Cannot import `autogluon`.

The `AutoGluonEstimator` is a thin wrapper for calling the `AutoGluon` package.
"""

class AutoGluonPredictor(Predictor):

    def __init__(self, model: TimeSeriesPredictor, prediction_length: int, freq: str, lead_time: int = 0) -> None:
        super().__init__(prediction_length, freq, lead_time)

        if not AUTOGLUON_IS_INSTALLED:
            raise ImportError(USAGE_MESSAGE)

        self.prediction_length = prediction_length
        self.freq = freq
        self.predictor = model
    
    def predict(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        **kwargs,
    ) -> Iterator[QuantileForecast]:
        data_frame = TimeSeriesDataFrame(dataset)
        outputs = self.predictor.predict(data_frame)

        metas = outputs.index.values
        cancat_len = outputs.shape[0]
        assert cancat_len % self.prediction_length == 0
        ts_num = cancat_len // self.prediction_length

        # resault wraper
        colums = outputs.columns[1:]
        for i in range(ts_num):
            cur_val = outputs.values[i * self.prediction_length : (i + 1) * self.prediction_length, 1:].T
            meta = metas[i * self.prediction_length : (i + 1) * self.prediction_length]
            yield QuantileForecast(
                forecast_arrays=cur_val,
                start_date=meta[0][1],
                freq=self.freq,
                forecast_keys=colums,
                item_id=meta[0][0])

    def leaderboard(
        self,
        dataset: Dataset,
        **kwargs,
    ):
        warnings.filterwarnings("ignore")
        data_frame = TimeSeriesDataFrame(dataset)
        model_path = os.getenv("SM_MODEL_DIR") or Path.home() / "models"
        leaderboard = self.predictor.leaderboard(data_frame)
        leaderboard.to_csv(Path.joinpath(Path(model_path), 'leaderboard.csv'))
        print('leaderboard has been saved at:', Path.joinpath(Path(model_path), 'leaderboard.csv'))

    def deserialize(cls, path: Path, **kwargs) -> "Predictor":
        predictor = TimeSeriesPredictor.load(cls, path)  # type: ignore
        file = path / "metadata.pickle"
        with file.open("r") as f:
            meta = json.load(f)
        return AutoGluonPredictor(model=predictor,
            freq=meta["freq"], prediction_length=meta["prediction_length"]
        )

    def serialize(self, path: Path) -> None:
        self.predictor.save()
        original_path = self.predictor._learner.path
        copy_tree(original_path, str(path))
        print(f"Saved trained model to {path}")
        file = path / "metadata.pickle"
        with file.open("w") as f:
            json.dump(
                {
                    "freq": self.freq,
                    "prediction_length": self.prediction_length,
                },
                f,
            )
