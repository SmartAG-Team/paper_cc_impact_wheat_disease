import numpy as np
import pandas as pd


def test_bank_selection_learns_training_physics_and_replays_without_query_outcomes():
    from models.monitoring.process_establishment_experiment import fit_bank, predict_bank
    rng = np.random.default_rng(22)
    bank = rng.uniform(.02, .98, (3, 70, 3, 8))
    physics = np.array([[.01, 1, 20], [.1, 1, 20], [.001, 1, 20]])
    train = pd.DataFrame(dict(source='BASF', leaf_index=0, observed_percent=100*bank[1, :, 0, 0]))
    fitted = fit_bank(bank, physics, train, np.ones(70))
    assert fitted['models']['process_selected']['physics_index'] == 1
    q = train.drop(columns='observed_percent')
    p = predict_bank(fitted, bank, q)
    assert np.mean((p['process_selected']-train.observed_percent)**2) < 1e-12
    assert np.mean((p['process_generic']-train.observed_percent)**2) > 1
    assert len(fitted['losses']) == 6

