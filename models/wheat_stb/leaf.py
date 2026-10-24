"""Engine adapter for the stage-anchored scalar leaf component."""

from model.seasonal_septoria.leaf_phenology import LeafCanopyModel as StageLeafModel
from .config import LeafParameters


class LeafCanopyModel:
    def __init__(self, parameters: LeafParameters):
        self.parameters = parameters
        self._model = StageLeafModel(dict(parameters.stage_thresholds),
            rank_spacing_units=parameters.rank_spacing_units,
            juvenile_policy=parameters.juvenile_policy)

    def calc_rates(self, current_accumulation, current_temperature, forcing_valid=True):
        return self._model.calc_rates(current_accumulation, current_temperature, forcing_valid)

    def integrate(self, rates):
        return self._model.integrate(rates)

    def snapshot(self):
        return self._model.snapshot()

    def restore(self, snapshot):
        self._model.restore(snapshot)
