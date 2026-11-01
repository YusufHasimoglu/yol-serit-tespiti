"""UFLD V2'nin ONNX-sonrası anahtar nokta ve ego-şerit seçim testleri."""
import numpy as np

from lanevision.engine import MODES
from lanevision.ufld import _row_candidates, choose_ego_pair


def _outputs():
    # [izgara, satır, şerit] biçimindeki küçük sentetik UFLD çıktısı.
    loc=np.full((20, 12, 4), -6., np.float32)
    exist=np.full((2, 12, 4), -5., np.float32)
    exist[0]=3.  # Varsayılan: çizgi yok.
    for row in range(12):
        # Orta kameralı ego şeridinin iki sınırı: x yaklaşık 35 ve 165.
        for lane, grid in ((1, 4 + row // 6), (2, 15 - row // 6)):
            exist[1, row, lane]=8.
            exist[0, row, lane]=-4.
            loc[grid, row, lane]=9.
    return loc, exist


def test_ufld_row_outputs_decode_to_two_curves():
    loc,exist=_outputs()
    candidates=_row_candidates(loc,exist,200,120)
    assert len(candidates)==2
    assert all(item['support']==12 for item in candidates)
    assert all(len(item['curve'])>=16 for item in candidates)


def test_ufld_chooses_pair_that_brackets_camera():
    loc,exist=_outputs()
    pair=choose_ego_pair(_row_candidates(loc,exist,200,120),200,120)
    assert pair is not None
    left,right,score=pair
    assert score>.4
    assert np.polyval(left['coeffs'],112)<100<np.polyval(right['coeffs'],112)


def test_ufld_is_an_explicit_user_selectable_mode():
    assert 'ufld' in MODES
    assert 'UFLD V2' in MODES['ufld']
