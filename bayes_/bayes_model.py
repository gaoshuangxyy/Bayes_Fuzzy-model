from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator,MaximumLikelihoodEstimator,BayesianEstimator
from pgmpy.inference import VariableElimination
from data_.model_1 import result

import pandas as pd
#导入数据
data = pd.read_csv('data.csv')
# pre_data = pd.read_csv(r'C:\Users\GS\Desktop\load_data\data_\model_1_result.csv')
pre_data = result

'''data中的1:0比例是18:1'''

#存在边
# eg：('X8', 'S8')代表存在从X8  to   S8的一条边
model = BayesianModel([('X8', 'S8'), ('X9', 'S9'), ('X1', 'S10'), ('X1', 'S12')
                       , ('X2', 'S10'), ('X2', 'S11'), ('X2', 'S13'), ('X3', 'S14')
                       , ('X4', 'S15'), ('X5', 'S16'), ('X6', 'S17'), ('X6', 'S7')
                       , ('X7', 'S7'), ('S10', 'S1'), ('S11', 'S1'), ('S12', 'S2')
                       , ('S13', 'S2'), ('X2', 'S3'), ('S14', 'S4'), ('S15', 'S4')
                       , ('S16', 'S4'), ('S17', 'S4'), ('S8', 'S5'), ('S9', 'S5')
                       , ('S1', 'S6'), ('S2', 'S6'), ('S3', 'S6'), ('S4', 'S6')
                       , ('S5', 'T'), ('S6', 'T'), ('S7', 'T')])


pe = ParameterEstimator(model, data)
# print("\n", pe.state_counts('S1'))

'''对模型和数据进行   极大似然估计  train'''
mle = MaximumLikelihoodEstimator(model, data)

# print("\n", mle.estimate_cpd('S1'))
# print("\n", mle.estimate_cpd('T'))  # 在fruit和size的条件下，tasty的概率分布

mle.get_parameters()
model.fit(data, estimator=MaximumLikelihoodEstimator)
#查看各个节点之间的概率分布
'''
print(model.get_cpds('S1'))
print(model.get_cpds('S7'))
print(model.get_cpds('T'))
'''


'''变量估计'''
infer = VariableElimination(model)
#输出infer的数据类型
# print(type(infer))



'''打印各个节点的概率'''
# for i in infer.query(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9'
#                       , 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10'
#                       , 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'T']).values():
#     print(i)
create = infer.query(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9'
                      , 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10'
                      , 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'T'])

# print('大，香蕉是美味的概率:\n', infer.query(['tasty'], evidence={'fruit': 1, 'size': 0})['tasty'])  # 大，香蕉是否美味的概率


'''贝叶斯估计'''
esy = BayesianEstimator(model, data)

# print(esy.estimate_cpd('T', prior_type='BDeu', equivalent_sample_size=10))

model2 = BayesianModel([('X8', 'S8'), ('X9', 'S9'), ('X1', 'S10'), ('X1', 'S12')
                       , ('X2', 'S10'), ('X2', 'S11'), ('X2', 'S13'), ('X3', 'S14')
                       , ('X4', 'S15'), ('X5', 'S16'), ('X6', 'S17'), ('X6', 'S7')
                       , ('X7', 'S7'), ('S10', 'S1'), ('S11', 'S1'), ('S12', 'S2')
                       , ('S13', 'S2'), ('X2', 'S3'), ('S14', 'S4'), ('S15', 'S4')
                       , ('S16', 'S4'), ('S17', 'S4'), ('S8', 'S5'), ('S9', 'S5')
                       , ('S1', 'S6'), ('S2', 'S6'), ('S3', 'S6'), ('S4', 'S6')
                       , ('S5', 'T'), ('S6', 'T'), ('S7', 'T')])
model2.fit(data, estimator=BayesianEstimator)
#输出各个节点之间的概率问题
#
# for i in model2.get_cpds():
#     print(i.variable)

infer2 = VariableElimination(model2)

'''   其中的“S4”代表我们需要求的概率，如果需要求多个，用逗号隔开
    evidence代cpds(表各个数据异常是否存在'''
result = infer2.query(['S4'],evidence = {"X1":1,"X2":1,"X3":1,"X4":1,"X5":1,"X6":1
                                         ,"X7":1,"X8":1,"X9":1})
# print(result)
# print(infer2.query(['tasty'], evidence={'fruit': 1, 'size': 0}))
pre_result = model2.predict_probability(pre_data)
# pre_result = pre_result[:,8:]
# print(pre_result.head())
end_result = pd.DataFrame(pre_result,columns=['S1_0.0','S2_0.0','S3_0.0','S4_0.0','S5_0.0','S6_0.0','S7_0.0','S8_0.0',
                                'S9_0.0','S10_0.0','S11_0.0','S12_0.0','S13_0.0','S14_0.0','S15_0.0','S16_0.0',
                                'S17_0.0'])
# print('故障的类型及概率：\n', end_result)
end_result.to_csv('result_pre.csv',index=None)



