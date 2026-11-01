import cv2
import numpy as np
import pytest
from lanevision.scene import SceneProfile,analyze_scene
from lanevision.scenario_data import generate_scenario

@pytest.mark.parametrize('scenario',['duz_gunduz','viraj_gunduz','duz_gece','viraj_gece','yagmurlu_gece'])
def test_scene_analyzer_returns_safe_profile(scenario):
    img,_=generate_scenario(900006,scenario)
    p=analyze_scene(img)
    assert isinstance(p,SceneProfile)
    assert p.lighting in ('gece','gunduz','alacakaranlik')
    assert p.geometry in ('duz','viraj','belirsiz')
    assert 0<=p.brightness<=255

def test_invalid_scene_rejected():
    with pytest.raises(ValueError):analyze_scene(np.zeros((30,30,3),np.uint8))

def test_night_and_day_illumination_can_be_distinguished():
    day=np.full((180,320,3),180,np.uint8)
    night=np.full((180,320,3),25,np.uint8)
    assert analyze_scene(day).lighting=='gunduz'
    assert analyze_scene(night).lighting=='gece'