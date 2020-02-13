# coding=UTF-8
import sys
import time
import pandas as pd
import os
from functools import reduce
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.exceptions import ConvergenceWarning
from joblib import dump
from sklearn import metrics

class StandardizeFile:

    all_input_data = []
    all_output_data = []
    input_df = 0
    output_df = 0

    def __init__(self, file_directory):

        self.file_directory = file_directory

        self.cv=5
        self.test_proportion=0.3

        self.solver_param='lbfgs'
        self.activation_param='identity'

        self.hidden_layer_sizes=[20,30,40,50,60]
        self.max_iter=[200,500,800,1100,1400]

        self.best_estimator={}
        self.params=[]

    # **************数据文件处理过程********

    def input_normally(self, filename):
        data = pd.read_table(self.file_directory + filename,
                             header=None,
                             encoding='gb2312',
                             sep='/s+',
                             engine='python',
                             names=['A'],
                             skiprows=79,
                             skipfooter=3,
                             )

        df_one_list = []
        for i in range(0, len(data), 2):
            row1_data = data['A'][i].split()
            row2_data = data['A'][i + 1].split()
            df_list = []
            df_list.append(float(row1_data[0]))
            df_list.append(float(row1_data[1]))
            df_list.append(float(row1_data[2]))
            df_list.append(float(row2_data[0]))
            df_one_list.append(df_list)

        StandardizeFile.all_input_data.append(df_one_list)
        StandardizeFile.input_df = pd.DataFrame(StandardizeFile.all_input_data)

    def output_normally(self, filename):
        data = pd.read_table(self.file_directory + filename,
                             header=None,
                             encoding='gb2312',
                             engine='python',
                             names=['A'],
                             skiprows=4,
                             skipfooter=0,
                             )
        df_one_list = []
        for i in range(0, len(data), 1):
            rows_data_str = data['A'][i].split()
            rows_data_float = list(map(float, rows_data_str))
            df_one_list.append(rows_data_float)

        StandardizeFile.all_output_data.append(df_one_list)
        StandardizeFile.output_df = pd.DataFrame(StandardizeFile.all_output_data)

    def standardize_file(self):
        for upar_file in [x for x in os.listdir(self.file_directory)]:
            if upar_file.split('.')[-1] == 'IN_NOSCALE_IATM1_dn':
                if os.path.exists(self.file_directory + upar_file + '.out'):
                    print('Process input: %s' % upar_file)
                    self.input_normally(upar_file)
                    self.output_normally(upar_file + '.out')

                else:
                    print('file: ' + upar_file + ' don\'t have relevant output file !' )

            else:
                print('file: ' + upar_file + ' is not formal !')

# *************************************************************

    # **************训练过程****************

    def set_param(self,cv,test_proportion,solver_param,activation_param,hidden_layer_sizes,max_iter):

        self.cv=cv
        self.test_proportion=test_proportion

        self.solver_param=solver_param
        self.activation_param=activation_param

        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter

        return True

    def processing_parameters(self):

       if len(self.hidden_layer_sizes)==len(self.max_iter):
             params1=list(zip(self.hidden_layer_sizes,))
             for i in range(len(self.hidden_layer_sizes)):
                temporary_dict = {}
                temporary_dict['hidden_layer_sizes'] = params1[i]
                temporary_dict['max_iter'] = self.max_iter[i]
                self.params.append(temporary_dict)
             return True
       else:
           print("参数不匹配")
           return False

    def save_model(self):

        try:

            clf=self.best_estimator
            end_time=time.strftime("%Y%m%d%H%M%S", time.localtime())
            file_name="model_{}_{}_{}_{}.m".format(end_time,self.cv,
                                          clf.get_params()['hidden_layer_sizes'],
                                                   clf.get_params()['max_iter'])
            dump(clf, file_name)
            return True
        except Exception as e:
            print("模型保存失败", e)

    def get_model_papam(self):

        clf=self.best_estimator

        # 获取此估计量的参数。
        model_para = {}
        index = 0
        model_para['迭代次数'] = clf.n_iter_
        model_para['网络层数'] = clf.n_layers_
        model_para['输出个数'] = clf.n_outputs_
        model_para['输出激活函数'] = clf.out_activation_
        model_para['模型参数'] = clf.get_params()

        #网络层偏移量
        for p in clf.intercepts_:
            index += 1
            b = '第{}层网络层偏移量:'.format(index)
            model_para[b] = p

        #网络层权重矩阵
        index = 0
        for w in clf.coefs_:
            index += 1
            a = '第{}层网络层权重矩阵:'.format(index)
            model_para[a] = w

        a = str(model_para)

        #定义文件名
        end_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        file_name = "model_{}_{}_{}_{}.json".format(end_time, self.cv,
                                                 clf.get_params()['hidden_layer_sizes'],
                                                 clf.get_params()['max_iter'])
        with open(file_name, 'w') as f:
            f.write(a + '\n')

    def train_model(self, input_dataset, output_dataset,output_features='Relative_humidity'):

        features_dict={"height":0,"pressure":1,"temperature":2,"Relative_humidity":3}

        try:
            best_clf = None
            best_score = sys.float_info.max
            test_scores = []

            X_tr = []
            Y_tr = []

            # 合并D.F列形成输入数据
            for j in range(output_dataset.shape[0]):
                  a = [output_dataset.iloc[j:j+1, n].values for n in range(output_dataset.shape[1])]
                  X_tr.append(list(a[i][0][2] for i in range(output_dataset.shape[1])))

            # 合并D.F列形成输出数据
            # valid_parameter=0:height，1：pressure，2：temperature（K），3：Relative_humidity
            for j in range(input_dataset.shape[0]):
                  a = [input_dataset.iloc[j:j+1, n].values for n in range(input_dataset.shape[1])]
                  Y_tr.append(list(a[i][0][features_dict[output_features]] for i in range(input_dataset.shape[1])))

            X_tr = np.array(X_tr)
            Y_tr = np.array(Y_tr)

            sScaler = StandardScaler()
            X_Scaled = sScaler.fit_transform(X_tr)

            c_v = ShuffleSplit(n_splits=self.cv, random_state=0, test_size=self.test_proportion)

            for train_index, test_index in c_v.split(X_Scaled):

                for param in self.params:

                    # create neural network using MLPRegressor
                    clf = MLPRegressor(solver=self.solver_param, activation=self.activation_param, alpha=1e-2,
                                       random_state=0, tol=1e-2,
                                       **param)

                    X_train, X_test = X_Scaled[train_index], X_Scaled[test_index]
                    y_train, y_test = Y_tr[train_index], Y_tr[test_index]

                    # 某些参数组合将无法收敛，因此此处将其忽略
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                                module="sklearn")
                        # 训练分类器
                        clf.fit(X_train, y_train)

                    # 测试集上进行预测
                    pre_test_y = clf.predict(X_test)

                    # 输出每折测试集上的mae

                    # uniform_average含义

                    test_score = metrics.mean_squared_error(y_test, pre_test_y, multioutput='uniform_average')
                    test_scores.append(test_score)

                    print("mean absolute error:",
                          metrics.mean_squared_error(y_test, pre_test_y, multioutput='uniform_average'))

                    # compare score of the n models and get the best one
                    if test_score < best_score:
                        best_score = test_score
                        best_clf = clf

            if best_clf != None:
                self.best_estimator = best_clf
                self.save_model()
                self.get_model_papam()
                print("训练完成，最小的均方误差为:%s。" % best_score)
                return True
            else:
                print("训练未完成")
                return False
        except Exception as e:
            print("训练过程出现异常", e)
            return False

if __name__ == '__main__':

    new_object = StandardizeFile(file_directory='in/')
    new_object.standardize_file()

    # new_object.set_param()
    new_object.processing_parameters()
    new_object.train_model(new_object.input_df,new_object.output_df)