import sys
import os
# 将项目的 src 目录添加到 sys.path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import pytest
from sensor_calib.calibrators.pinhole_camera_calibrator import PinholeCameraCalibrator

@pytest.fixture
def calibrator():
    # 传入初始化所需参数
    configs = {"sensor_name": "front_camera","sensor_type": "pinhole_camera"}
    return PinholeCameraCalibrator(configs)
def test_pinhole_camera_calibrator(calibrator):
    # 测试run函数
    calibrator.run()
    # 测试save函数
    calibrator.save()