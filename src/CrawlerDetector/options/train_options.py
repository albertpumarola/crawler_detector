from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--n_threads_train', default=4, type=int, help='# threads for loading data')
        self._parser.add_argument('--print_freq_s', type=int, default=5, help='frequency of showing training results on console')
        self._parser.add_argument('--display_freq_s', type=int, default=30, help='frequency [s] of showing training results on screen')
        self._parser.add_argument('--save_latest_freq_s', type=int, default=3600, help='frequency of saving the latest results')
        self._parser.add_argument('--nepochs_no_decay', type=int, default=20, help='# of epochs at starting learning rate')
        self._parser.add_argument('--nepochs_decay', type=int, default=10, help='# of epochs to linearly decay learning rate to zero')
        self._parser.add_argument('--weight_decay', type=float, default=0, help='weight decay lambda value')

        self.is_train = True

