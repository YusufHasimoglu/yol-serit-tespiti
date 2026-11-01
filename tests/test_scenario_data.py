import numpy as np
import pytest
from lanevision.scenario_data import SCENARIOS,generate_scenario

@pytest.mark.parametrize('scenario',SCENARIOS)
def test_synthetic_night_curve_scenarios_have_valid_masks(scenario):
    img,masks=generate_scenario(100,scenario)
    assert img.shape==(144,256,3)
    assert masks.shape==(144,256,2)
    assert masks.dtype==np.uint8
    assert np.count_nonzero(masks[:,:,0])>20
    assert np.count_nonzero(masks[:,:,1])>20
    img2,masks2=generate_scenario(100,scenario)
    assert np.array_equal(img,img2)
    assert np.array_equal(masks,masks2)


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):generate_scenario(1,'bilinmeyen')