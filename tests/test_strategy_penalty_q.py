import numpy as np
import pandas as pd

from bot.strategy import penalty_to_penalty_q


def test_penalty_q_matches_penalty_without_quantization():
    penalty = pd.Series([0.0, 0.01, 0.19, 0.21, 1.03])
    penalty_q = penalty_to_penalty_q(penalty)

    assert np.allclose(penalty_q.to_numpy(), penalty.to_numpy(), atol=0.0, rtol=0.0)
