import json
from pathlib import Path
import cv2
import numpy as np
from lanevision.processing import process_image, process_video, read_image
from lanevision.sample_data import draw_scene


def test_image_and_stages(tmp_path):
    origin = tmp_path / 'yol_çizgisi.png'
    output = tmp_path / 'işlenmiş.png'
    cv2.imwrite(str(tmp_path / 'temp.png'), draw_scene())
    origin.write_bytes((tmp_path / 'temp.png').read_bytes())
    result = process_image(origin, output, stages_dir=tmp_path/'adımlar')
    assert result.both_found
    assert read_image(output).shape == (720, 1280, 3)
    assert len(list((tmp_path/'adımlar').glob('*.png'))) == 8


def test_video_output_and_reports(tmp_path):
    source, output = tmp_path/'input.avi', tmp_path/'sonuç.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 9.0, (640, 360))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(cv2.resize(draw_scene(), (640, 360)))
    writer.release()
    summary = process_video(source, output)
    assert summary.frames == 4
    assert summary.detected_both > 0
    assert output.is_file() and output.stat().st_size > 0
    summary_path = output.with_name('sonuç_ozet.json')
    assert summary_path.exists()
    loaded = json.loads(summary_path.read_text(encoding='utf-8'))
    assert loaded['video']['islenen_kare'] == 4
    assert output.with_name('sonuç_istatistik.csv').exists()


def test_same_input_output_video_rejected(tmp_path):
    source = tmp_path/'input.mp4'
    source.write_bytes(b'not a real video')
    try:
        process_video(source, source)
    except ValueError as exc:
        assert 'farklı' in str(exc)
    else:
        assert False, 'Expected ValueError'
