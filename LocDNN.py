import numpy as np
import os
import glob # check if file exists using regex
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import configparser
import sys

import os
import sys
my_modules_dir = os.path.expanduser('~/my_modules')
sys.path.append(os.path.join(my_modules_dir,'basic_tools/basic_tools'))
import TFData
from get_fpath import get_fpath

class LocDNN(object):
    def __init__(self,config_fpath=None,is_load_model=False,gpu_index=0):

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # check input
        if config_fpath is None and meta_fpath is None:
            raise Exception('neither config_fpath or meta_fpath is given')

        # constant member variables
        self.fea_len = 37
        self.eps = 1e-20
        self.n_azi = 37

        # create graph and session
        self._graph = tf.Graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = '{}'.format(gpu_index)
        self._sess = tf.compat.v1.Session(graph=self._graph,config=config)

        self._load_config(config_fpath)
        self._build_model()

        # whether to load existed model
        self.is_load_model = is_load_model
        if self.is_load_model:
            self._load_model(self.model_dir)
        else:
            existed_meta_fnames = glob.glob(''.join([self.model_dir,'*.meta']))
            if len(existed_meta_fnames) > 0:
                if input('existed models are found\
                      {}\
                      if overwrite[y/n]'.format(existed_meta_fnames)) == 'y':
                    os.removedirs(self.model_dir)
                else:
                    return
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

        # open text file for logging
        self._log_file = open(os.path.join(self.model_dir,'log.txt'),'a')


    def _add_log(self,log_info):
        self._log_file.write(log_info)
        self._log_file.write('\n')
        if self.is_print_log:
            print(log_info)


    def _load_config(self,config_fpath):
        """
        """
        if config_fpath is not None:
            config = configparser.ConfigParser()
            config.read(config_fpath)
            # model settings
            self.model_dir = config['model']['model_dir']
            self.azi_num = np.int16(config['model']['azi_num'])
            self.frame_len = np.int16(config['model']['frame_len'])
            self.is_norm = config['model']['is_norm']=='True'
            self.norm_params_fpath = config['model']['norm_params_fpath']
            self.is_dropout = config['model']['is_dropout']=='True'
            # training settings
            self.batch_size= np.int16(config['train']['batch_size'])
            self.max_epoch = np.int16(config['train']['max_epoch'])
            self.is_print_log = config['train']['is_print_log']=='True'
            self.train_set_dir = config['train']['train_set_dir'].split(';')
            self.valid_set_dir = config['train']['valid_set_dir'].split(';')
            print('train_set{} \n valid_set{}'.format(self.train_set_dir,
                                                      self.valid_set_dir))


    def _load_model(self,model_dir):
        """load model"""
        if not os.path.exists(model_dir):
            raise Exception('no model exists in {}'.format(model_dir))

        with self._graph.as_default():
            # restore model
            saver = tf.compat.v1.train.Saver()
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self._sess,ckpt.model_checkpoint_path)


    def _train_record_init(self):
        """"""
        try:
            # load record vars from file
            record_info=np.load(os.path.join(saved_model_dir,
                                          'train_record.npz'))
            # training record on train data
            cost_record_train=record_info['cost_record_train']
            rmse_record_train=record_info['rmse_record_train']
            # training record on test data
            cost_record_valid=record_info['cost_record_valid']
            rmse_record_valid=record_info['rmse_record_valid']
            # other record vars
            lr_value=record_info['lr_value']
            min_valid_cost=record_info['min_valid_cost']
            best_epoch=record_info['best_epoch']
            last_epoch=np.nonzero(cost_record_train)[0][-1]
            self._add_log('load training record vars')
        except Exception as e:
            # record vars initialization
            cost_record_train=np.zeros(self.max_epoch)
            rmse_record_train=np.zeros(self.max_epoch)
            cost_record_valid=np.zeros(self.max_epoch)
            rmse_record_valid=np.zeros(self.max_epoch)
            lr_value=1e-3
            min_valid_cost=np.infty
            best_epoch=0
            last_epoch=-1 # train start from last_epoch+1
            self._add_log('initialize training record vars')
        return [cost_record_train,rmse_record_train,
                cost_record_valid,rmse_record_valid,
                lr_value,min_valid_cost,best_epoch,last_epoch]


    def _build_model(self):
        """build graph
        """
        with self._graph.as_default():
            # core of model
            x = tf.compat.v1.placeholder(shape=[None,self.fea_len],
                                         dtype=tf.float32,
                                         name='x')#

            layer1_fcn = tf.keras.layers.Dense(units=1024,
                                           activation=tf.nn.sigmoid)
            layer1_out = layer1_fcn(x)

            layer2_fcn = tf.keras.layers.Dense(units=1024,
                                           activation=tf.nn.sigmoid)
            layer2_out = layer2_fcn(layer1_out)

            layer3_dropout = tf.keras.layers.Dropout(rate=0.5)
            layer3_out = layer3_dropout(layer2_out)

            layer4_fcn =tf.keras.layers.Dense(units=37,
                                              activation=tf.nn.softmax)
            y_est = layer4_fcn(layer3_out)

            # for train
            y = tf.compat.v1.placeholder(shape=[None,self.azi_num],
                                         dtype=tf.float32)#

            # measurements
            cost = self._cal_cross_entropy(y_est,y)
            azi_rmse = self._cal_azi_rmse(y_est,y)
            # cp = self._cal_cp(y_est,y)
            # rmse = self._cal_rmse(y_est,y)

            # for train
            lr = tf.compat.v1.placeholder(tf.float32, shape=[])
            opt_step = tf.compat.v1.train.AdamOptimizer(lr).minimize(cost)

            # data pipline
            coord = tf.train.Coordinator()
            n_batch_queue = 20
            xy_len = self.fea_len+self.n_azi
            train_tfdata = TFData.TFData([None,self.fea_len],
                                         [None,self.n_azi],
                                         sess=self._sess,
                                         batch_size=self.batch_size,
                                         n_batch_queue=n_batch_queue,
                                         coord=coord,
                                         file_reader=self._file_reader)

            # initialization graph
            init = tf.compat.v1.global_variables_initializer()
            self._sess.run(init)
            #
            self._x = x
            self._y_est = y_est
            self._y = y
            #
            self._cost = cost
            self._azi_rmse =azi_rmse

            self._lr = lr
            self._opt_step = opt_step
            #
            self._coord = coord
            self._tfdata = train_tfdata


    def _cal_cross_entropy(self,y_est,y):
        """cross entrophy"""
        cross_entropy = -tf.reduce_mean(\
                            tf.reduce_sum(tf.multiply(y,
                                                  tf.math.log(y_est+self.eps)),
                                          axis=1))
        return cross_entropy


    def _cal_azi_rmse(self,y_est,y):
        azi_est = tf.argmax(y_est,axis=1)
        azi = tf.argmax(y,axis=1)
        diff = tf.cast(azi_est-azi,tf.float32)
        return tf.sqrt(tf.reduce_mean(tf.pow(diff,2)))


    # def _cal_rmse(self,y_est,y):
    #     """Mean absolute error
    #     Args:
    #         y_est: model output
    #         y: labels
    #     Returns:
    #         mean absolute error, one float number
    #     """
    #     azi_est = tf.argmax(y_est,axis=1)
    #     azi = tf.argmax(y,axis=1)
    #     diff = tf.cast(azi_est-azi,tf.float32)
    #     rmse = tf.sqrt(tf.reduce_mean(tf.pow(diff,2)))
    #     return rmse


    # def _cal_cp(self,y_est,y):
    #     """calculate mean correct percentage
    #     Args:
    #         y_est: model output
    #         y: labels
    #     Returns:
    #         correct percentage, one float number
    #     """
    #     azi_est = tf.argmax(y_est,axis=1)
    #     azi = tf.argmax(y,axis=1)
    #     equality = tf.equal(azi_est,azi)
    #     cp = tf.reduce_mean(tf.cast(equality,tf.float32),name='cp')
    #     return cp


    def _pack_xy(self,x,y):
        xy = np.concatenate((x,y),axis=1)
        return xy


    def _unpack_xy(self,xy):
        x = xy[:,:self.fea_len]
        y = xy[:,self.fea_len:]
        return [x,y]


    def _file_reader(self,dataset_dir):
        """Read spectrum files under given directory
        Args:
            dataset_dir:
        Returns:
            sample generator, [samples,label_onehots]
        """
        if isinstance(dataset_dir,list):
            dir_list = dataset_dir
        else:
            dir_list = [dataset_dir]

        mean,std = np.load(self.norm_params_fpath)

        npy_fpaths = []
        for sub_set_dir in dir_list:
            fea_fpath_sub = get_fpath(sub_set_dir,'.npy')
            npy_fpaths.extend([os.path.join(sub_set_dir,item)
                                            for item in fea_fpath_sub])
        np.random.shuffle(npy_fpaths)# randomize files order

        if len(npy_fpaths) < 1:
            raise Exception('folder is empty: {}'.format(dataset_dir))

        for fpath in npy_fpaths:
            gcc_phats = np.load(fpath)
            gcc_phats_norm = np.divide(gcc_phats-mean,std)

            fname,_ = os.path.basename(fpath).split('.')
            azi,_ = map(int,fname.split('_'))
            onehot_labels=np.zeros((gcc_phats_norm.shape[0],self.azi_num))
            onehot_labels[:,azi]=1

            yield [gcc_phats_norm,onehot_labels]


    def train_model(self):
        """Train model either from scratch or from last epoch
        """
        # n_sample = 0
        # for var_list in self._file_reader(self.train_set_dir):
        #     n_sample = n_sample+var_list[0].shape[0]
        # print(n_sample)
        # return
        with self._graph.as_default():
            # load or initialize record vars
            [cost_record_train,rmse_record_train,
            cost_record_valid,rmse_record_valid,
            lr_value,min_valid_cost,
            best_epoch,last_epoch] = self._train_record_init()

            saver = tf.compat.v1.train.Saver()
            thread_list = []
            print('start traning')
            for epoch in range(last_epoch+1,self.max_epoch):
                t_start = time.time()
                # start pipeline thread, feed data to model for train
                self._tfdata.start(file_dir=self.train_set_dir,
                                         n_thread=1,
                                         is_repeat=False)
                n_batch = 0
                t_start_iter = time.time()
                while not self._tfdata.query_if_finish():
                    x,y = self._sess.run(self._tfdata.var_batch)
                    self._sess.run(self._opt_step,
                                   feed_dict={self._x:x,
                                              self._y:y,
                                              self._lr:lr_value})
               # model test
                [cost_record_train[epoch],
                rmse_record_train[epoch]] = self.evaluate(self.train_set_dir)

                [cost_record_valid[epoch],
                rmse_record_valid[epoch]] = self.evaluate(self.valid_set_dir)

                # write to log
                epoch_time = time.time()-t_start
                self._add_log('epoch:{} lr:{} time:{:.2f}\n'.format(epoch,lr_value,epoch_time))
                self._add_log('\t train ')
                log_template = '\t cost:{:.2f} rmse:{:.2f}\n'
                self._add_log(log_template.format(cost_record_train[epoch],
                                                  rmse_record_train[epoch]))
                self._add_log('\t valid ')
                self._add_log(log_template.format(cost_record_valid[epoch],
                                                  rmse_record_valid[epoch]))

                # find new optimal
                if min_valid_cost>cost_record_valid[epoch]:
                    self._add_log('find new optimal\n')
                    best_epoch = epoch
                    min_valid_cost=cost_record_valid[epoch]
                    saver.save(self._sess,os.path.join(self.model_dir,'model'),
                               global_step=epoch)

                    np.savez(os.path.join(self.model_dir,'train_record'),
                             cost_record_train=cost_record_train,
                             rmse_record_train=rmse_record_train,
                             cost_record_valid=cost_record_valid,
                             rmse_record_valid=rmse_record_valid,
                             lr=lr_value,
                             best_epoch=best_epoch,
                             min_valid_cost=min_valid_cost)

                # early stop
                # no new optimal in 5-epoch
                if epoch-best_epoch>5:
                    self._add_log('early stop %f\n'%min_valid_cost)
                    break

                # adaptive learning rate
                if epoch > 2:
                    if cost_record_train[epoch] > \
                            np.min(cost_record_train[epoch-2:epoch]):
                        lr_value = lr_value*.2

            self._coord.request_stop()
            [self._coord.join(thread) for thread in thread_list]
            self._log_file.close()

            if True:
                # performance on train set
                fig,axs = plt.subplots(1,2)
                axs[0].plot(cost_record_train,label='train')
                axs[0].plot(cost_record_valid,label='valid')
                axs[0].set_title('cross_entrophy')

                axs[1].plot(rmse_record_train,label='train')
                axs[1].plot(rmse_record_valid,label='valid')
                axs[1].set_title('RMSE')
                axs[1].legend()
                #
                plt.tight_layout()
                fig_path = os.path.join(self.model_dir,'train_curve.png')
                plt.savefig(fig_path)


    def predict(self,x):
        """ predict sound position of given data x
        """
        y_est = self._sess.run(self._y_est,feed_dict={self._x:x})
        return y_est


    def evaluate(self,set_dir):
        cost_all = 0.
        rmse_all = 0.
        n_sample_all = 0.

        self._tfdata.start(file_dir=self.train_set_dir,
                                 is_repeat=False)
        while not self._tfdata.query_if_finish():
            x,y = self._sess.run(self._tfdata.var_batch)
            n_sample_tmp = x.shape[0]
            cost_tmp,rmse_tmp = self._sess.run([self._cost,self._azi_rmse],
                                                feed_dict={self._x:x,
                                                           self._y:y})
            #
            n_sample_all = n_sample_all+n_sample_tmp
            cost_all = cost_all+n_sample_tmp*cost_tmp
            rmse_all = rmse_all+n_sample_tmp*(rmse_tmp**2)
        #
        cost_all = cost_all/n_sample_all
        rmse_all = np.sqrt(rmse_all/n_sample_all)

        return [cost_all,rmse_all]


    def evaluate_chunk(self,data_set_dir,chunk_size=25):
        """ Evaluate model on given data_set
        Args:
            data_set_dir:
        Returns:
            [rmse_chunk,cp_chunk,rmse_frame,cp_frame]
        """
        print(data_set_dir)
        xy_generator = self._file_reader(data_set_dir)

        rmse_chunk = 0.
        n_chunk = 0

        for xy in xy_generator:
            x,y = self._unpck_xy(xy)

            sample_num = x.shape[0]
            azi_true = np.argmax(y[0])

            y_est = self.predict(x)
            for sample_i in range(0,sample_num-chunk_size+1):
                azi_est_chunk = np.argmax(np.mean(y_est[sample_i:\
                                                         sample_i+chunk_size],
                                                  axis=0))
                rmse_chunk = rmse_chunk + (azi_est_chunk-azi_true)**2
                n_chunk = n_chunk+1

        rmse_chunk = np.sqrt(rmse_chunk/n_chunk)*5
        return rmse_chunk

    def evaluate_chunk(self,data_set_dir,chunk_size=25):
        """ Evaluate model on given data_set, only for loc
        Args:
            data_set_dir:
        Returns:
        """

        rmse_chunk = 0.
        n_chunk = 0

        for x,y in self._file_reader(dataset_dir=data_set_dir):

            sample_num = x.shape[0]
            azi_true = np.argmax(y[0])

            y_est = self.predict(x)
            for sample_i in range(0,sample_num-chunk_size+1):
                azi_est_chunk = np.argmax(np.mean(y_est[sample_i:\
                                                         sample_i+chunk_size],
                                                  axis=0))
                rmse_chunk = rmse_chunk + (azi_est_chunk-azi_true)**2
                n_chunk = n_chunk+1

        rmse_chunk = np.sqrt(rmse_chunk/n_chunk)*5
        return rmse_chunk
