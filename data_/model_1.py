import numpy as np
from outliers import smirnov_grubbs as grubbs
import pandas as pd
import copy

num_need_repeat = 4
data = pd.read_excel('b.xlsx',header=None,names=['b'],sheet_name='b')
sliding = 300
sliding_num = 1
jump = 50

for i in range(10):
    data.loc[i] = 10.0
img = [0]*len(data)


'''提前将数据进行处理，find重复、缺失跳变、为零的数据标记为异常数据'''
'''error_data作为返回值，当erro_data==1时代表存在以上异常情况中的一种或多种'''
def proform(data):

    error_data = [0] * len(data)
    data_repeat = data['b'].diff(2)
    df = data.loc[data['b'].diff(2) == 0]
    df_index = list(df.index)
    for item in df_index:
        error_data[i] = 1
    # 1.2缺失
    data_missing = data[data['b'].isnull() == True]
    for item in data_missing:
        error_data[i] = 1
    # 1.3 and 1.4 判断跳变,为零
    for iterm_jump in range(len(data) - 1):
        if ((abs(data.iloc[iterm_jump + 1].values - data.iloc[iterm_jump].values) >= jump) or (
                data.iloc[iterm_jump].values == 0.0)):
            error_data[iterm_jump] = 1
    return error_data

num_x1_miss,num_x2_miss = 5,20
num_repeat,num_zero = 5,5


'''l_data是标记数据是否发生故障的list'''

'''数据中断(瞬时)'''
def abnormal_x1(l_data,data):
    for item_data in range(len(data)):
        if (l_data[item_data]!=0):
            for miss in range(num_x1_miss):
                if(data.iloc[item_data+miss].isnull().values==[False]):break;
                if miss == num_x1_miss-1:l_data[item_data] = (l_data[item_data]-1)*10+2;
    return l_data
'''数据中断(长期)'''
def abnormal_x2(l_data,data):
    for item_data in range(len(data)):
        if (l_data[item_data] != 0):
            for miss in range(num_x2_miss):
                if (data.iloc[item_data+miss].isnull().values == [False]): break;
                if miss == num_x2_miss - 1: l_data[item_data] = (l_data[item_data]-1)*10+3;
    return l_data
'''数据重复 '''
def abnormal_x3(l_data,data):
    error_data = [0] * len(data)
    #data_repeat = data['b'].diff(2)
    df = data.loc[data['b'].diff(2) == 0]
    df_index = list(df.index)
    # print(df_index)
    for item in df_index:
        error_data[item] = 1
    # print(error_data)
    if len(df) >= 4:
        for num_in_df_index in range(len(df_index)-3):
            if df_index[num_in_df_index]==df_index[num_in_df_index+1]-1==df_index[num_in_df_index+2]-2==df_index[num_in_df_index+3]-3:
                pass
            else :
                df_index[num_in_df_index] = -1
    df_index = df_index[:len(df_index)-3]
    df_index_deepcopy = copy.deepcopy(df_index)
    for df_index_num in df_index:
        if df_index_num==-1:
            df_index_deepcopy.remove(-1)
    if len(df_index_deepcopy)==0:
        print('没有重复值')
    else :
        for item in df_index_deepcopy:
            l_data[item] = 4
        # print(df_index_deepcopy)
    return l_data
    pass
'''数据固定偏移---检测值与实际值的差别'''
def abnormal_x4(l_data,data):
    pass
'''数据为零'''
def abnormal_x5(l_data,data):
    for item_data in range(len(data)):
        if (l_data[item_data] != 0):
            for miss in range(num_zero):
                if (data.iloc[item_data+miss].values != [0]): break;
                if miss == num_zero - 1: l_data[item_data] = (l_data[item_data]-1)*10+6;
    return l_data
    pass


def func_jump_error(value):
    #value是单维的数据


    # return list(temp)
    data_copy = value.copy()
    data_series = pd.Series(value['b'].values, index=value['b'].index)
    temp = set(grubbs.test(data_series, alpha=0.01))
    # print(temp)
    # all_right_data = func_jump_error((data_series))
    all_right_data = list(temp)
    for item_right in all_right_data:
        data_copy = data_copy[~data_copy['b'].isin([item_right])]
    # print(data_copy)  # data_copy 表示离群点
    # jump_error = set(value) - set(list(grubbs.test(value,alpha=0.01)))
    # return list(jump_error)
    return data_copy


num_of_outlier_to_dif_x8_x9 = 10


'''数据连续增长(降低)'''
def abnormal_x6_x8(l_data,data):
    D_value = []
    for item_x6 in range(len(data)-1):
        D_value.append(abs(float(data.iloc[item_x6].values)-float(data.iloc[item_x6+1].values)))
    data_x6 = set(grubbs.test(D_value, alpha=0.01))
    # print(temp)
    # all_right_data = func_jump_error((data_series))
    # all_right_data = list(temp)
    data_x6_remove = list(set(D_value).difference(data_x6))
    # return data_x6_remove
    if(len(data_x6_remove)>0):return 8
    else:return 6         #d但是我感觉这个只能适用于某一个范围，不能适用于整个窗口，至少不能存在一个就判定为抖动


def residal(data):#求残差
    sum = 0.0
    mean_ = data.mean()
    for residal_item in data:
        # print(type(float(residal_item)))
        sum+=(float(residal_item)-mean_)**2
    return sum
    pass
'''数据跳边'''
def abnormal_x6_x7(l_data,data):
    data = data['b']
    residal_list,standard_num,standard_num_up = [],0,1
    standard = residal(data)
    residal_list.append(residal(data))
    for item_x6_x7 in range(1,len(data)-1):
        k_temp = residal(data[0:item_x6_x7])+residal(data[item_x6_x7:len(data-1)])
        residal_list.append(k_temp)
        if k_temp>3*standard:
            standard_num+=1
    residal_list.append(residal(data))
    if standard_num_up>1:
        return 7
    else:
        return 6
'''数据抖动'''
def abnormal_x8(l_data,data):
    pass
'''离群点'''
def abnormal_x8_x9(l_data,data):
    '''
    data_copy = data.copy()
    data_series = pd.Series(data['b'].values,index = data['b'].index)
    all_right_data = func_jump_error((data_series))
    for item_right in all_right_data:
        data_copy = data_copy[~data_copy['b'].isin([item_right])]
    print(data_copy)#data_copy 表示离群点
    '''
    out = func_jump_error(data)
    if len(out)<num_of_outlier_to_dif_x8_x9:
        return 9
    else:
        return 8
    pass


'''error_x1到5是list类型，value(元素）-1，代表发生的故障类型'''
# if __name__ == '__main__':
for item in range(1,sliding_num+1):#滑动窗口大小设置，正式算法接近完成的时候将对data进行限定。
    error_data = proform(data)
    # error_data_x_5 = abnormal_x5(error_data,data)
    # print(error_data_x_5)
    result = [0.0]*9
    error_x1 = abnormal_x1(error_data, data)
    if 2 in error_x1:
        result[0] = 1.0
    else:
        result[0] = 0.0
    error_x2 = abnormal_x2(error_data, data)
    if 3 in error_x1:
        result[1] = 1.0
    else:
        result[1] = 0.0
    error_x3 = abnormal_x3(error_data, data)
    if 4 in error_x1:
        result[2] = 1.0
    else:
        result[2] = 0.0
    error_x5 = abnormal_x5(error_data, data)
    if 6 in error_x1:
        result[4] = 1.0
    else:
            result[4] = 0.0
        # print(error_x5)
        #
        #
    alt = abnormal_x6_x8(error_data, data)
    if alt==8:
        result[7] = 1.0
    else:
        result[5] = 1.0
    alt = abnormal_x6_x7(error_data,data)
    if alt==7:
        result[6] = 1.0
    else:
        result[5] = 1.0
    alt = abnormal_x8_x9(error_data,data)
    if alt==9:
        result[8] = 1.0
    else:
        result[7] = 1.0
    result = pd.DataFrame(result,index = ['X1','X2','X3','X4','X5','X6','X7','X8','X9']).T
        # print(result)
    # result.to_csv('model_1_result.csv',index=None)
    # print('数据中存在的数据异常模式:\n', result)


















































    '''
    error_data = [0]*len(data)
    data_repeat = data['b'].diff(2)
    df = data.loc[data['b'].diff(2)==0]
    df_index = list(df.index)
    for item in df_index:
        error_data[i] = 1
    #1.2缺失
    data_missing = data[data['b'].isnull()==True]
    for item in data_missing:
        error_data[i] = 1
    #1.3判断跳变,为零
    for iterm_jump in range(len(data)-1):
        if((data.iloc[iterm_jump+1].values-data.iloc[iterm_jump].values>=jump) or(data.iloc[iterm_jump].values==0.0)):
            error_data[iterm_jump] = 1
    print(error_data)
    '''

    # if len(df)>=4:
    #     for num_in_df_index in range(len(df_index)-3):
    #         # print('kk', df_index[num_in_df_index],df_index[num_in_df_index]-1,df_index[num_in_df_index]-2,df_index[num_in_df_index]-3)
    #         if df_index[num_in_df_index]==df_index[num_in_df_index+1]-1==df_index[num_in_df_index+2]-2==df_index[num_in_df_index+3]-3:
    #             pass
    #         else :
    #             df_index[num_in_df_index] = -1
    # df_index = df_index[:len(df_index)-3]
    # df_index_deepcopy = copy.deepcopy(df_index)
    # for df_index_num in df_index:
    #     if df_index_num==-1:
    #         df_index_deepcopy.remove(-1)
    #
    # if len(df_index_deepcopy)==0:
    #     print('没有重复值')
    # else :print(df_index_deepcopy)

    #1.2判断缺失
    # print(data)
    # data_null = data.isnull()
    # print(data_null)
    # data_true = data[data_null['b']==True]
    # print(data_true)
    # #2、判断数据流异常模式X1-9
    # pass