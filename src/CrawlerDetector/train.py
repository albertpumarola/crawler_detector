import time
from options.train_options import TrainOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from models.models import ModelsFactory
from util.tb_visualizer import TBVisualizer
from collections import OrderedDict
import os


class Train:
    def __init__(self):
        self._opt = TrainOptions().parse()
        data_loader_train = CustomDatasetDataLoader(self._opt, is_for_train=True)
        data_loader_test = CustomDatasetDataLoader(self._opt, is_for_train=False)

        self._dataset_train = data_loader_train.load_data()
        self._dataset_test = data_loader_test.load_data()

        self._dataset_train_size = len(data_loader_train)
        self._dataset_test_size = len(data_loader_test)
        print('#train images = %d' % self._dataset_train_size)
        print('#test images = %d' % self._dataset_test_size)

        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._tb_visualizer = TBVisualizer(self._opt)

        self._train()

    def _train(self):
        self._total_steps = self._opt.load_epoch * self._dataset_train_size
        self._iters_per_epoch = self._dataset_train_size / self._opt.batch_size
        self._last_display_time = None
        self._last_save_latest_time = None
        self._last_print_time = time.time()

        i_epoch = None
        for i_epoch in range(self._opt.load_epoch + 1, self._opt.nepochs_no_decay + self._opt.nepochs_decay + 1):
            # train epoch
            self._train_epoch(i_epoch)

            # update learning rate
            if i_epoch > self._opt.nepochs_no_decay:
                self._model.update_learning_rate()

        # save last epoch
        if i_epoch is not None:
            self._model.save(i_epoch)

    def _train_epoch(self, i_epoch):
        epoch_iter = 0
        self._model.set_train()
        iter_start_time = time.time()
        iter_read_time = 0
        iter_procs_time = 0
        num_iters_time = 0
        for i_train_batch, train_batch in enumerate(self._dataset_train):
            iter_read_time += (time.time() - iter_start_time) / self._opt.batch_size
            iter_after_read_time = time.time()

            # display flags
            do_visuals = self._last_display_time is None or time.time() - self._last_display_time > self._opt.display_freq_s
            do_print_terminal = time.time() - self._last_print_time > self._opt.print_freq_s or do_visuals

            # train model
            self._model.set_input(train_batch)
            self._model.forward(do_visuals)
            self._model.optimize_parameters()

            # update epoch info
            self._total_steps += self._opt.batch_size
            epoch_iter += self._opt.batch_size
            iter_procs_time += (time.time() - iter_after_read_time) / self._opt.batch_size
            num_iters_time += 1

            # display terminal
            if do_print_terminal:
                iter_read_time /= num_iters_time
                iter_procs_time /= num_iters_time
                self._display_terminal(iter_read_time, iter_procs_time, i_epoch, i_train_batch, do_visuals)
                self._last_print_time = time.time()
                iter_read_time = 0
                iter_procs_time = 0
                num_iters_time = 0

            # display visualizer
            if do_visuals:
                self._display_visualizer_train(self._total_steps)
                self._display_visualizer_val(i_epoch, self._total_steps)
                self._last_display_time = time.time()

            # save model
            if self._last_save_latest_time is None or time.time() - self._last_save_latest_time > self._opt.save_latest_freq_s:
                print('saving the latest model (epoch %d, total_steps %d)' % (i_epoch, self._total_steps))
                self._model.save(i_epoch)
                self._last_save_latest_time = time.time()

            iter_start_time = time.time()

    def _display_terminal(self, iter_read_time, iter_procs_time, i_epoch, i_train_batch, visuals_flag):
        errors = self._model.get_current_errors()
        self._tb_visualizer.print_current_train_errors(i_epoch, i_train_batch, self._iters_per_epoch, errors, iter_read_time, iter_procs_time, visuals_flag)

    def _display_visualizer_train(self, total_steps):
        self._tb_visualizer.display_current_results(self._model.get_last_saved_visuals(), total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_errors(), total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_scalars(), total_steps, is_train=True)

    def _display_visualizer_val(self, i_epoch, total_steps):
        val_start_time = time.time()

        # set model to eval
        self._model.set_eval()

        # evaluate self._opt.num_iters_validate epochs
        val_errors = OrderedDict()
        i_val_batch = 0
        for i_val_batch, val_batch in enumerate(self._dataset_test):

            # evaluate model
            self._model.set_input(val_batch)
            self._model.forward(keep_data_for_visuals=(i_val_batch == 0))
            errors = self._model.get_current_errors()

            # store current batch errors
            for k, v in errors.iteritems():
                if k in val_errors:
                    val_errors[k] += v
                else:
                    val_errors[k] = v

            break

        # normalize errors
        for k in val_errors.iterkeys():
            val_errors[k] /= (i_val_batch+1)

        # visualize
        t = (time.time() - val_start_time)
        self._tb_visualizer.print_current_validate_errors(i_epoch, val_errors, t)
        self._tb_visualizer.plot_scalars(val_errors, total_steps, is_train=False)
        self._tb_visualizer.display_current_results(self._model.get_last_saved_visuals(), total_steps, is_train=False)

        # set model back to train
        self._model.set_train()


if __name__ == "__main__":
    Train()
