import numpy as np
import warnings
warnings.filterwarnings("ignore")

def s_weights(n,array):
    judge_array = np.array(array).reshape(n, n)
    # print(type(judge_array))
    '''计算指标集权重'''
    #矩阵的特征值和特征向量
    eig1, eig2 = np.linalg.eig(judge_array)
    # print("eig is :")
    # print(eig1)
    # print("eig matrix is :")
    # print(eig2)
    #一致性检验
    '''n=12,ri=1.54'''
    dict_RI = {'1':0,'2':0,'3':0.52,'4':0.89,'5':1.12,'6':1.26,'7':1.36,'8':1.41,
               '9':1.46,'10':1.49,'11':1.52,'12':1.54,'13':1.56,'14':1.58,'15':1.59,}

    ri = dict_RI[str(n)]
    ci = (eig1[0] - n) / (n - 1)
    # print('ci is:',ci)
    cr = ci / ri
    # print("cr is:",cr)
    if cr < 0.1:
        # print('一致性检验通过')
        # sum_num = 0
        mat_weight = np.zeros((n))
        for i in range(n):
            mat_weight[i] = eig2[i][0]
        # print('指标重要程度权重I:\n', mat_weight)
        # 归一化处理
        guiyi_mat_weight = mat_weight / np.sum(mat_weight)
        # print('归一化后指标重要程度权重I：\n', guiyi_mat_weight)
        # print(type(guiyi_mat_weight))
        return mat_weight, guiyi_mat_weight
    else:
        print('一致性检验不通过，调整指标重要性')

def W_weight(array,P):
    '''结合故障概率，得出指标风险权重集ω={ω1,, ω2, ω3, ω4, ω5, ω6, ω7, ω8, ω9， ω10, ω11, ω12}'''
    #假设S集中的每种故障的概率我们可从故障数据中得出为P

    # print(rate)
    # P = rate[columns].mode().values[0]
    # rate = list(rate['rate_right'])
    # P = np.array([1/3*rate[1],rate[5]+rate[6],rate[7],rate[8],1/2*rate[0],
    #             1/3*rate[1],1/2*rate[0],1/3*rate[1],rate[2],rate[3],rate[4],rate[5]])
    #W为指标风险权重集ω,并归一化
    W = P*array
    guiyi_W_weight = W / np.sum(W)
    # print('归一化后的指 v标风险权重集w:\n',guiyi_W)
    return guiyi_W_weight

def lishudu_func(x):
    '''隶属度函数'''
    x = list(x)
    par = np.linspace(min(x), max(x), num=4)
    y_list = []
    for i in range(len(x)):
        if x[i] >= par[0] and x[i] < par[1]:
            y = (x[i] - par[0]) / (par[1] - par[0])
            y_list.append(y)
        elif x[i] >= par[1] and x[i] < par[2]:
            y = 1
            y_list.append(y)
        else:
            y = (max(x)-x[i]) / (max(x) - par[2])
            y_list.append(y)
    return y_list

#if __name__ == '__main__':
'''构建指标集S={S3，S7，S8，S9，S10，S11， S12，S13，S14，S15，S16，S17}的判断矩阵'''
S_mat_origin = [1, 5, 7, 7, 3, 3, 5, 5, 6, 6, 6, 6,
                  1 / 5, 1, 3, 3, 1 / 5, 1 / 5, 1 / 6, 1 / 6, 1 / 5, 1 / 5, 1 / 5, 1 / 5,
                  1 / 7, 1 / 3, 1, 1, 1 / 5, 1 / 5, 1 / 6, 1 / 6, 1 / 7, 1 / 7, 1 / 7, 1 / 7,
                  1 / 7, 1 / 3, 1, 1, 1 / 5, 1 / 5, 1 / 6, 1 / 6, 1 / 7, 1 / 7, 1 / 7, 1 / 7,
                  1 / 3, 5, 5, 5, 1, 1, 2, 2, 1, 1, 1, 1,
                  1 / 3, 5, 5, 5, 1, 1, 2, 2, 1, 1, 1, 1,
                  1 / 5, 6, 6, 6, 1 / 2, 1 / 2, 1, 1, 1 / 5, 1 / 5, 1 / 5, 1 / 5,
                  1 / 5, 6, 6, 6, 1 / 2, 1 / 2, 1, 1, 3, 3, 3, 3,
                  1 / 6, 5, 7, 7, 1, 1, 5, 1 / 3, 1, 1, 1, 1,
                  1 / 6, 5, 7, 7, 1, 1, 5, 1 / 3, 1, 1, 1, 1,
                  1 / 6, 5, 7, 7, 1, 1, 5, 1 / 3, 1, 1, 1, 1,
                  1 / 6, 5, 7, 7, 1, 1, 5, 1 / 3, 1, 1, 1, 1]
'''构建每个指标在三种状态下V={正常，异常，故障}的判读矩阵'''
s3_mat_origin =[1, 5, 7, 1/5, 1, 3, 1/7, 1/3, 1]
s7_mat_origin =[1, 1/5, 1/7, 5, 1, 1/3, 7, 3, 1]
s8_mat_origin = [1, 5, 3, 1/5, 1, 1/2, 1/3, 2, 1]
s9_mat_origin = [1, 2, 3, 1/5, 1, 1/2, 1/3, 2, 1]
s10_mat_origin = [1, 3, 5, 1/3, 1, 2, 1/5, 1/2, 1]
s11_mat_origin = [1, 1/4, 1/3, 4, 1, 2, 3, 1/2, 1]
s12_mat_origin = [1, 1/7, 1/5, 7, 1, 3, 5, 1/3, 1]
s13_mat_origin = [1, 1/4, 1/6, 4, 1, 1/3, 6, 3, 1]
s14_mat_origin = [1, 1/5, 1/3, 5, 1, 2, 3, 1/2, 1]
s15_mat_origin = [1, 1/4, 1/3, 4, 1, 2, 3, 1/2, 1]
s16_mat_origin = [1, 1/3, 1/2, 3, 1, 2, 2, 1/2, 1]
s17_mat_origin = [1, 1/3, 1/2, 3, 1, 1, 2, 1, 1]

s3_, s3 = s_weights(3, s3_mat_origin)
s7_, s7 = s_weights(3, s7_mat_origin)
s8_, s8 = s_weights(3, s8_mat_origin)
s9_, s9 = s_weights(3, s9_mat_origin)
s10_, s10 = s_weights(3, s10_mat_origin)
s11_, s11 = s_weights(3, s11_mat_origin)
s12_, s12 = s_weights(3, s12_mat_origin)
s13_, s13 = s_weights(3, s13_mat_origin)
s14_, s14 = s_weights(3, s14_mat_origin)
s15_, s15 = s_weights(3, s15_mat_origin)
s16_, s16 = s_weights(3, s16_mat_origin)
s17_, s17 = s_weights(3, s17_mat_origin)
    
s_mat_weight, s_guiyi_mat_weight = s_weights(12, S_mat_origin)
# print(s_mat_weight)
# print(s_mat_weight.shape)
guiyi_weight = np.concatenate((s3, s7, s8, s9, s10, s11,
                             s12,s13,s14,s15,s16,s17), axis=0).reshape(12,3)
guiyi_weight_1 = np.concatenate((s3, s7, s8, s9, s10, s11,
                                   s12, s13, s14, s15, s16, s17), axis=0)
weight_1 = np.concatenate((s3_, s7_, s8_, s9_, s10_, s11_,
                             s12_,s13_,s14_,s15_,s16_,s17_), axis=0).reshape(12,3)
#指标权重归一化后带入隶属度函数后的权重
weight = np.array(lishudu_func(guiyi_weight_1)).reshape(12,3)
# w_weight = W_weight(s_mat_weight,p)
# print('归一化后的指标得分：\n', guiyi_weight)
# print('指标权重归一化后带入隶属度函数后的权重;\n',weight)

    # for i in range(len(S_Probability)):
    #     # print(S_Probability.loc[i].values)
    #     w_weight = W_weight(s_mat_weight,S_Probability.loc[i].values)
    #     result = np.dot(w_weight,weight)
    #     print('评判结果：', result)
    # print('归一化前的指标得分：\n', weight_1)
    # print('归一化后的指标风险权重集W:\n',w_weight)




