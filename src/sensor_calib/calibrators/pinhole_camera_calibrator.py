from typing import List
from dataclasses import dataclass, field

from .base_calibrator import BaseCalibrator

@dataclass
class PinholeCameraCalibratorParam:
    name: str = "default_cam"
    image_files: List[str] = field(default_factory=list)
    save_dir: str = "./results"
    
class PinholeCameraCalibrator(BaseCalibrator):
    def __init__(self, params: PinholeCameraCalibratorParam):
        super().__init__(params)

    def save(self, save_dir: str = "./results"):
        pass

    def _calibrate_intrinsics(self):
        pass

    def _apply_intrinsics(self,data):
        pass
    def _calibrate_extrinsics(self):
        # 针孔相机一般外参在多相机场景或与外部传感器配准时标定
        pass
    def _apply_extrinsics(self,data):
        pass
