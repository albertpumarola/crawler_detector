from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--output_estimation_dir', type=str, default='/media/apumarola/ssd_dataset/object_detector/test', help='')
        self.is_train = False
