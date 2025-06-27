class BaseCalibrator:
    def __init__(self, params):
        self.params = params
        self.results = {}
    def save(self, save_dir: str = "./results"):
        raise NotImplementedError
    def get_results(self):
        return self.results
    def run(self):
        if self.params.get("intrinsics", {}).get("calibrate", False):
            self._calibrate_intrinsics()
        if self.params.get("extrinsics", {}).get("calibrate", False):
            self._calibrate_extrinsics()

    def _calibrate_intrinsics(self):
        raise NotImplementedError
    def _apply_intrinsics(self,data):
        raise NotImplementedError
    def _calibrate_extrinsics(self):
        raise NotImplementedError
    def _apply_extrinsics(self,data):
        raise NotImplementedError

