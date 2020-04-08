from fault_.fault_describe import X, S
from data_.model_1 import result
from bayes_.bayes_model import end_result
from fuzzy_.fuzzy_model import weight, W_weight, s_mat_weight
import numpy as np
import pandas as pd

if __name__ == '__main__':
    '''模型一的输出'''
    print('\n模型一输出数据中存在的数据异常模式:\n', result)
    print('数据中存在的异常模式：')
    for i in result.T.index:
        # print(result[i].values[0])
        if result[i].values[0] == 1.0:
            print(X[i])
    '''模型二的输出'''
    print('\n模型二输出的故障类型及概率：\n', end_result)
    # print('最大值：', max(end_result.values[0]))
    max_value = max(end_result.values[0])
    for i in end_result.columns:
        if float(max_value) == float(end_result[i]):
            print('故障可能为：',S[i])
    '''模型三的输出'''
    for i in range(len(end_result)):
        # print(S_Probability.loc[i].values)
        columns = ['S3_0.0', 'S7_0.0', 'S8_0.0', 'S9_0.0', 'S10_0.0', 'S11_0.0', 'S12_0.0',
                   'S13_0.0', 'S14_0.0', 'S15_0.0', 'S16_0.0', 'S17_0.0']
        end_result = end_result[columns]
        w_weight = W_weight(s_mat_weight, end_result.loc[i].values)
        result = np.dot(w_weight, weight)
        result_ = pd.DataFrame(result, index=['正常', '异常', '故障'], columns=['result'])
        print('\n模型三的评判结果：\n', result_)
        # print(result_.index())
        print('设备状态：', str(result_[result_.result == float(max(result_.values))].index.tolist()[0]))