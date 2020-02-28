import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import configparser
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class LocDNN(object):
    def __init__(self, file_reader, config_fpath=None, gpu_index=0):

        # constant member variables
        self.epsilon = 1e-20

        self._file_reader = file_reader

        # create graph and session
        self._graph = tf.Graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = '{}'.format(gpu_index)
        self._sess = tf.compat.v1.Session(graph=self._graph, config=config)

        self._load_config(config_fpath)
        self._build_model()

    def _add_log(self, log_info):
        self._log_file.write(log_info)
        self._log_file.write('\n')
        if self.is_print_log:
            print(log_info)

    def _load_config(self, config_fpath):
        """
        """
        if config_fpath is not None:
            config = configparser.ConfigParser()
            config.read(config_fpath)
            # model settings
            self.model_dir = config['model']['model_dir']
            self.n_azi = np.int16(config['model']['n_azi'])
            self.fea_len = np.int16(config['model']['fea_len'])
            self.is_norm = config['model']['is_norm'] == 'True'
            self.norm_coef_fpath = config['model']['norm_coef_fpath']
            self.is_dropout = config['model']['is_dropout'] == 'True'
            # training settings
            self.batch_size = np.int16(config['train']['batch_size'])
            self.max_epoch = np.int16(config['train']['max_epoch'])
            self.is_print_log = config['train']['is_print_log'] == 'True'
            self.train_set_dir = config['train']['train_set_dir'].split(';')
            self.valid_set_dir = config['train']['valid_set_dir'].split(';')
            if self.valid_set_dir[0] == '':
                self.valid_set_dir = None

            print('Train set:')
            [print('\t{}'.format(item)) for item in self.train_set_dir]

            print('Valid set:')
            [print('\t{}'.format(item)) for item in self.valid_set_dir]

    def _build_model(self):
        """build graph
        """
        with self._graph.as_default():
            # core of model
            x = tf.compat.v1.placeholder(shape=[None, self.fea_len],
                                         dtype=tf.float32,
                                         name='x')

            layer1_fcn = tf.keras.layers.Dense(units=1024,
                                               activation=tf.nn.sigmoid)
            layer1_out = layer1_fcn(x)

            layer2_fcn = tf.keras.layers.Dense(units=1024,
                                               activation=tf.nn.sigmoid)
            layer2_out = layer2_fcn(layer1_out)

            layer3_dropout = tf.keras.layers.Dropout(rate=0.5)
            layer3_out = layer3_dropout(layer2_out)

            layer4_fcn = tf.keras.layers.Dense(units=37,
                                               activation=tf.nn.softmax)
            y_est = layer4_fcn(layer3_out)

            # for train
            y = tf.compat.v1.placeholder(shape=[None, self.n_azi],
                                         dtype=tf.float32)

            # measurements
            cost = self._cal_cross_entropy(y_est, y)
            azi_rmse = self._cal_azi_rmse(y_est, y)

            # for train
            lr = tf.compat.v1.placeholder(tf.float32, shape=[])
            opt_step = tf.compat.v1.train.AdamOptimizer(lr).minimize(cost)

            # initialization graph
            init = tf.compat.v1.global_variables_initializer()
            self._sess.run(init)
            #
            self._x = x
            self._y_est = y_est
            self._y = y
            #
            self._cost = cost
            self._azi_rmse = azi_rmse

            self._lr = lr
            self._opt_step = opt_step

    def _cal_cross_entropy(self, y_est, y):
        cross_entropy = -tf.reduce_mean(
                            tf.reduce_sum(
                                tf.multiply(
                                    y, tf.math.log(y_est+self.epsilon)),
                                axis=1))
        return cross_entropy

    def _cal_mse(self, y_est, y):
        rmse = tf.reduce_mean(tf.reduce_sum((y-y_est)**2, axis=1))
        return rmse

    def _cal_azi_rmse(self, y_est, y):
        azi_est = tf.argmax(y_est, axis=1)
        azi = tf.argmax(y, axis=1)
        diff = tf.cast(azi_est - azi, tf.float32)
        return tf.sqrt(tf.reduce_mean(tf.pow(diff, 2)))

    def _cal_cp(self, y_est, y):
        equality = tf.equal(tf.argmax(y_est, axis=1), tf.argmax(y, axis=1))
        cp = tf.reduce_mean(tf.cast(equality, tf.float32))
        return cp

    def _train_record_init(self, model_dir=None, is_load_model=False):
        """"""
        if is_load_model:
            # load record vars from file
            record_info = np.load(os.path.join(self.model_dir,
                                               'train_record.npz'))
            cost_record_valid = record_info['cost_record_valid']
            azi_rmse_record_valid = record_info['azi_rmse_record_valid']
            # other record vars
            lr_value = record_info['lr_value']
            min_valid_cost = record_info['min_valid_cost']
            best_epoch = record_info['best_epoch']
            last_epoch = np.nonzero(cost_record_valid)[0][-1]
            self._add_log('load training record vars')
        else:
            # record vars initialization
            cost_record_valid = np.zeros(self.max_epoch)
            azi_rmse_record_valid = np.zeros(self.max_epoch)
            lr_value = 1e-3
            min_valid_cost = np.infty
            best_epoch = 0
            last_epoch = -1  # train start from last_epoch+1
            self._add_log('initialize training record vars')
        return [cost_record_valid, azi_rmse_record_valid,
                lr_value, min_valid_cost, best_epoch, last_epoch]

    def load_model(self, model_dir):
        """load model"""
        if not os.path.exists(model_dir):
            raise Exception('no model exists in {}'.format(model_dir))

        with self._graph.as_default():
            # restore model
            saver = tf.compat.v1.train.Saver()
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self._sess, ckpt.model_checkpoint_path)
            print(f'load model from {model_dir}')

    def train_model(self, model_dir, is_load_model=False):
        """Train model either from scratch or from last epoch
        """

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # open text file for logging
        self._log_file = open(os.path.join(self.model_dir, 'log.txt'), 'a')

        # whether to load existed model
        if is_load_model:
            self.load_model(model_dir)

        with self._graph.as_default():
            # load or initialize record vars
            [cost_record_valid, azi_rmse_record_valid,
             lr_value, min_valid_cost,
             best_epoch, last_epoch] = self._train_record_init(model_dir,
                                                               is_load_model)

            saver = tf.compat.v1.train.Saver()
            print('start traning')
            for epoch in range(last_epoch+1, self.max_epoch):
                t_start = time.time()
                batch_generator = self._file_reader(self.train_set_dir,
                                                    self.norm_coef_fpath,
                                                    self.batch_size)
                for x, y in batch_generator:
                    self._sess.run(self._opt_step,
                                   feed_dict={self._x: x,
                                              self._y: y,
                                              self._lr: lr_value})
                # model test
                [cost_record_valid[epoch],
                 azi_rmse_record_valid[epoch]] = self.evaluate(
                                                    self.valid_set_dir)

                # write to log
                iter_time = time.time()-t_start
                self._add_log(' '.join((f'epoch:{epoch}',
                                        f'lr:{lr_value}',
                                        f'time:{iter_time:.2f}\n')))

                log_template = '\t cost:{:.2f} azi_rmse:{:.2f}\n'
                self._add_log('\t valid ')
                self._add_log(log_template.format(
                                                cost_record_valid[epoch],
                                                azi_rmse_record_valid[epoch]))

                #
                if min_valid_cost > cost_record_valid[epoch]:
                    self._add_log('find new optimal\n')
                    best_epoch = epoch
                    min_valid_cost = cost_record_valid[epoch]
                    saver.save(self._sess, os.path.join(self.model_dir,
                                                        'model'),
                               global_step=epoch)

                    # save record info
                    np.savez(os.path.join(self.model_dir, 'train_record'),
                             cost_record_valid=cost_record_valid,
                             azi_rmse_record_valid=azi_rmse_record_valid,
                             lr=lr_value,
                             best_epoch=best_epoch,
                             min_valid_cost=min_valid_cost)

                # early stop
                if epoch-best_epoch > 5:
                    print('early stop\n', min_valid_cost)
                    self._add_log('early stop{}\n'.format(min_valid_cost))
                    break

                # learning rate decay
                if epoch > 2:  # no better performance in 2 epoches
                    min_valid_cost_local = np.min(
                                        cost_record_valid[epoch-1:epoch+1])
                    if cost_record_valid[epoch-2] < min_valid_cost_local:
                        lr_value = lr_value*.2

            self._log_file.close()

            if True:
                fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True)
                ax[0].plot(cost_record_valid)
                ax[0].set_ylabel('cross entrophy')

                ax[1].plot(azi_rmse_record_valid)
                ax[1].set_ylabel('rmse(deg)')
                #
                fig_path = os.path.join(self.model_dir, 'train_curve.png')
                plt.savefig(fig_path)

    def predict(self, x):
        """Model output of x
        """
        y_est = self._sess.run(self._y_est, feed_dict={self._x: x})
        return y_est

    def evaluate(self, set_dir):
        cost_all = 0.
        rmse_all = 0.
        n_sample_all = 0.

        batch_generator = self._file_reader(set_dir, self.norm_coef_fpath) 
        for x, y in batch_generator:
            n_sample_tmp = x.shape[0]
            cost_tmp, rmse_tmp = self._sess.run([self._cost, self._azi_rmse],
                                                feed_dict={self._x: x,
                                                           self._y: y})
            #
            n_sample_all = n_sample_all+n_sample_tmp
            cost_all = cost_all+n_sample_tmp*cost_tmp
            rmse_all = rmse_all+n_sample_tmp*(rmse_tmp**2)
        #
        cost_all = cost_all/n_sample_all
        rmse_all = np.sqrt(rmse_all/n_sample_all)
        return [cost_all, rmse_all]

    def evaluate_chunk(self, data_set_dir, chunk_size=25):
        """ Evaluate model on given data_set
        Args:
            data_set_dir:
        Returns:
            [rmse_chunk,cp_chunk,rmse_frame,cp_frame]
        """
        rmse_chunk = 0.
        n_chunk = 0

        for x, y in self._file_reader(data_set_dir, self.norm_coef_fpath):
            n_sample = x.shape[0]
            azi_true = np.argmax(y[0])

            y_est = self.predict(x)

            for sample_i in range(0, n_sample-chunk_size+1):
                azi_est_chunk = np.argmax(
                                    np.mean(
                                        y_est[sample_i:sample_i+chunk_size],
                                        axis=0))
                rmse_chunk = rmse_chunk+(azi_est_chunk-azi_true)**2
                n_chunk = n_chunk+1

        rmse_chunk = np.sqrt(rmse_chunk/n_chunk)*5
        return rmse_chunk
