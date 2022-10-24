---
title: XGBoost实战
---
```python
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime
```

# 导入参数


```python
data = load_boston() # 以波士顿房价为训练集

x = data.data
y = data.target

Xtrain,Xtest,Ytrain,Ytest = TTS(x,y,test_size=0.3,random_state=420)
```

    
    

# 对模型进行训练


```python
reg = XGBR(n_estimators=100).fit(Xtrain,Ytrain) # 参数实例化
reg.predict(Xtest) # 模型接口

reg.score(Xtest,Ytest) # 模型评估，r^2
```




    0.9050988968414799




```python
MSE(Ytest,reg.predict(Xtest)) # 均方误差
```




    8.830916343629323




```python
reg.feature_importances_ #模型重要性负数,每一个特征值的权重
```




    array([0.01902167, 0.0042109 , 0.01478316, 0.00553537, 0.02222196,
           0.37914088, 0.01679686, 0.0469872 , 0.04073574, 0.05491759,
           0.06684221, 0.00869464, 0.3201119 ], dtype=float32)



# 交叉验证，使用没有训练的模型


```python
reg = XGBR(n_estimators=100)
CVS(reg,Xtrain,Ytrain,cv=5).mean() # 模型，数据，折数，得到交叉验证的评估指标 r^2
```




    0.7995062821902295




```python
CVS(reg,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean() #负均方误差
```




    -16.215644229762717



# 模型对比


```python
rfr = RFR(n_estimators=100) # 使用随机森林进行对比
CVS(rfr,Xtrain,Ytrain,cv=5).mean()
```




    0.7977853495049871




```python
lr = LinearR() # 线性回归
CVS(lr,Xtrain,Ytrain,cv=5).mean()
```




    0.6835070597278076




```python
reg = XGBR(n_estimators=10,verbosity=0) 
CVS(reg,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()
```




    -18.633733952333067



# 绘制学习曲线


```python
def plot_learning_curve(estimator,title, X, y, 
                        ax=None, # 选择子图
                        ylim=None, # 设置纵坐标的取值范围
                        cv=None, # 交叉验证
                        n_jobs=None # 设定索要使用的线程
                       ):
    
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y
                                                            ,shuffle=True
                                                            ,cv=cv
                                                            ,random_state=420
                                                            ,n_jobs=n_jobs)      
    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid() # 绘制网格，不是必须
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-'
            , color="r",label="Training score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-'
            , color="g",label="Test score")
    ax.legend(loc="best")
    return ax
```


```python
cv = KFold(n_splits=5,shuffle=True,random_state=42)# 交叉验证模式
```


```python
plot_learning_curve(XGBR(n_estimators=100,random_state=420),"XGB",Xtrain,Ytrain,ax=None,cv=cv)
plt.show()
```


    
![png](/images/xgb_1.png)
    


# 进行调参

## 绘制学习曲线观察n_estimators的影响
### 对10~1010进行计算


```python
axisx = range(10,1010,50)
rs = []
for i in axisx:
    reg = XGBR(n_estimators=i,random_state=420)
    rs.append(CVS(reg,Xtrain,Ytrain,cv=cv).mean())
print(axisx[rs.index(max(rs))],max(rs))
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()
```

    160 0.8320776498992342
    


    
![png](/images/xgb_2.png)
    


### 方差与泛化误差

泛化误差：E(f;D),方差：var,偏差：bias,噪声：ξ。<br>
$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$**E(f;D) = bias² + var + ξ²**
          


```python
axisx = range(50,1050,50)
rs = [] # r^2
var = [] # 方差
ge = [] # 可控部分泛化误差
for i in axisx:
    reg = XGBR(n_estimators=i,random_state=420)
    cvresult = CVS(reg,Xtrain,Ytrain,cv=cv)
    
    rs.append(cvresult.mean()) # 记录1-偏差,r^2   
    var.append(cvresult.var()) # 记录方差    
    ge.append((1 - cvresult.mean())**2+cvresult.var())# 计算泛化误差的可控部分

print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))]) #打印R2最高所对应的参数取值，并打印这个参数下的方差
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var)) #打印方差最低时对应的参数取值，并打印这个参数下的R2
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge)) #打印泛化误差可控部分的参数取值，并打印这个参数下的R^2，方差以及泛化误差的可控部分
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()
```

    100 0.8320924293483107 0.005344212126112929
    100 0.8320924293483107 0.005344212126112929
    100 0.8320924293483107 0.005344212126112929 0.03353716440826495
    


    
![png](/images/xgb_3.png)
    


### 细化学习曲线，找到最佳n_estimators


```python
axisx = range(50,200,10) # 将范围缩小
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=i,random_state=420)
    cvresult = CVS(reg,Xtrain,Ytrain,cv=cv)
    
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1 - cvresult.mean())**2+cvresult.var())
    
print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
rs = np.array(rs)
var = np.array(var)*0.01

# R^2线
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="black",label="XGB")

# 添加方差线
plt.plot(axisx,rs+var,c="red",linestyle='-.')
plt.plot(axisx,rs-var,c="red",linestyle='-.')
plt.legend()
plt.show()

# 绘制泛化误差可控部分
plt.figure(figsize=(20,5))
plt.plot(axisx,ge,c="gray",linestyle='-.')
plt.show()
```

    100 0.8320924293483107 0.005344212126112929
    100 0.8320924293483107 0.005344212126112929
    100 0.8320924293483107 0.005344212126112929 0.03353716440826495
    


    
![png](/images/xgb_4.png)
    



    
![png](/images/xgb_5.png)
    


得到最佳取值为100。

## 有放回随机抽样，重要参数subsample


 ## 了解有放回随机抽样<br>
$~~~~~~~$第一次随机从原始数据集中进行随机抽取，将其进行建模。把模型结果，即预测错误的样本反馈回原始数据集，增加其权重。然后循环此过程，不断修正之前判断错误的样本。一定程度上提升模型准度。<br>
$~~~~~~~$在sklearn和xgboost中，使用subsample作为随机抽样参数，范围为(0，1]，默认为一，为抽取占原数据比例。

### 进行学习曲线分析


```python
axisx = np.linspace(0.75,1,25)# 随机选取数值，多次调整，缩减范围
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=100,subsample=i,random_state=420)
    cvresult = CVS(reg,Xtrain,Ytrain,cv=cv)
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1 - cvresult.mean())**2+cvresult.var())
    
print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
rs = np.array(rs)
var = np.array(var)
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="black",label="XGB")
plt.plot(axisx,rs+var,c="red",linestyle='-.')
plt.plot(axisx,rs-var,c="red",linestyle='-.')
plt.legend()
plt.show()
```

    1.0 0.8320924293483107 0.005344212126112929
    0.9375 0.8213387927010547 0.0023009355462403113
    1.0 0.8320924293483107 0.005344212126112929 0.03353716440826495
    


    
![png](/images/xgb_6.png)
    



```python
reg = XGBR(n_estimators=100,subsample=0.9375,random_state=420).fit(Xtrain,Ytrain)
reg.score(Xtest,Ytest)
```




    0.9175041345374846




```python
MSE(Ytest,reg.predict(Xtest))
```




    7.676560781152187



显然优于没有设置时的模型。如果模型准确率降低则取消使用此参数。

### 将n_estimators和subsample一起进行训练取得最优解


```python
axisx_sub = np.linspace(0.75,1,25)
axisx_n_est = range(50,200,10)
rs = []
var = []
ge = []
for i in axisx_n_est:
    for j in axisx_sub:
        reg = XGBR(n_estimators=i,subsample=j,random_state=420,eta=0.1)
        cvresult = CVS(reg,Xtrain,Ytrain,cv=cv)
        rs.append(cvresult.mean())
        var.append(cvresult.var())
        ge.append((1 - cvresult.mean())**2+cvresult.var())
        
print(axisx_n_est[int(rs.index(max(rs))/25)],axisx_sub[rs.index(max(rs))%25],max(rs),var[rs.index(max(rs))])
print(axisx_n_est[int(var.index(min(var))/25)],axisx_sub[var.index(min(var))%25],rs[var.index(min(var))],min(var))
print(axisx_n_est[int(ge.index(min(ge))/25)],axisx_sub[ge.index(min(ge))%25],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
```

    160 0.9583333333333333 0.8432917907566264 0.004614845818885538
    50 0.9479166666666666 0.8339831443137825 0.003529311838690004
    190 0.9479166666666666 0.842287142171737 0.004044787817216516 0.028918133341574434
    

## 迭代决策树，重要参数eta<br>

参数eta为迭代决策树的步长。<br>
迭代树公式为：<br>
$~~~~~~~~~~~~~~~~~~~~~~~~~$$y_i^{k+1} = y_i^k + ŋf_{k+1}^{x_i}$<br>
eta即为ŋ，也称为学习率，取值范围为[0,1]。在xgboost中默认为0.3，在sklearn中为0.1。


```python
# 评分函数
def regassess(reg,Xtrain,Ytrain,cv,scoring = ["r^2"],show=True):
    score = []
    for i in range(len(scoring)):
        if show:
            print("{}:{:.2f}".format(scoring[i] #模型评估指标的名字
                                     ,CVS(reg
                                          ,Xtrain,Ytrain
                                          ,cv=cv,scoring=scoring[i]).mean()))
        score.append(CVS(reg,Xtrain,Ytrain,cv=cv,scoring=scoring[i]).mean())
    return score
```


```python
axisx = np.arange(0.05,1,0.05)
rs = []
te = []
for i in axisx:
    reg = XGBR(n_estimators=100,subsample=0.9375,random_state=420,learning_rate=i)
    score = regassess(reg,Xtrain,Ytrain,cv,scoring = ["r2","neg_mean_squared_error"],show=False)
    test = reg.fit(Xtrain,Ytrain).score(Xtest,Ytest)
    rs.append(score[0])
    te.append(test)
    
print(axisx[rs.index(max(rs))],max(rs))
plt.figure(figsize=(20,5))
plt.plot(axisx,te,c="gray",label="test")
plt.plot(axisx,rs,c="green",label="train")
plt.legend()
plt.show()
```

    0.35000000000000003 0.8398273910404873
    


    
![png](/images/xgb_7.png)
    



```python
reg = XGBR(n_estimators=100,subsample=0.9375,random_state=420,eta=0.35).fit(Xtrain,Ytrain)
reg.score(Xtest,Ytest)
```




    0.8948044049000868




```python
MSE(Ytest,reg.predict(Xtest))
```




    9.78885881330489



发现并没有提升，反而下降。eta一般作为模型训练时间的调整指标，如果模型没有错误，一般都会收敛，eta影响并不算大。根据训练经验，eta取值一般在0.1左右

# XGBoost<br> 
## 选择弱评估器：booster参数<br>
$~~~~~~~$在sklearn中为booster参数，在xgboost中为xgb_model。用于选择弱评估器，可以输入的值有：gbtree、gblinear和dart。**gbtree**为梯度提升树，**gblinear**为线性模型，**dart**为抛弃提升树，比梯度提升树有更好的放过拟合功能。在xgboost中必须用param传入参数。


```python
for booster in ["gbtree","gblinear","dart"]:
    reg = XGBR(n_estimators=190
               ,subsample=0.9479166666666666
               ,learning_rate=0.1
               ,random_state=420               
               ,booster=booster).fit(Xtrain,Ytrain)
    print(booster)
    print(reg.score(Xtest,Ytest))
```

    gbtree
    0.9199261332157886
    gblinear
    0.6518219184964559
    dart
    0.9261191829728914
    

## 目标函数：objective<br>
$~~~~~~~$在xgboost中使用目标函数来替代传统损失函数。xgb的目标函数可以写为：**损失函数+模型复杂度**。<br>
$~~~~~~~~~~~~~~~~~~~$ $Obj = \sum_{i=1}^m l(y_i , y_i') + \sum_{k=1}^K \Omega(f_k)$

$~~~~~~~$第一项为损失函数，用来评估模型准确度。第二项为模型复杂度和树结构复杂度有直接联系，运用此函数添加在损失函数之后，可以使模型在准确的同时用最少的运算时间。当两者都最小，则平衡了模型效果和工程能力，使XGBoost运算又快又准。<br>
$~~~~~~~$方差用于评估模型稳定性，偏差用于评估模型的准确率。对应此目标函数。第一项衡量的是偏差，第二项则衡量方差。所以此目标函数也表示方差偏差平衡，也就是泛化误差。<br><br>
$~~~~~~~$对于xgboost，可以选择第一项的使用函数。在xgboost中参数为obj(也可以写成object)，在sklearn中为objective。在**xgb.train()** 和**xgb.XGBClassifier()** 中默认为 **binary:logistic**，在**xgb.XGBRegressor()** 中默认为 **reg:linear** 。

常用的参数有：<br>

|输入|使用的损失函数|
|:-----:|:----:|
|reg:linear |使用线性回归的损失函数，均方误差，回归时使用 |
|binary:logistic|使用逻辑回归的损失函数，对数损失log_loss,二分类时使用 |
|binary:hinge |使用支持向量机的损失函数，Hinge Loss，二分类时使用 |
|multi:softmax |使用softmax损失函数，多分类时使用 | 

**现在推荐使用reg:squarederror来代替reg:linear。**<br>
$~~~~~~~$使用xgboost库来进行训练的流程：<br>
$xgb.DMatrix() -> param=\{\} -> bst=xgb.train(param) -> bst.predict()$ <br>
 $ 读取数据->设置参数->训练模型->预测结果 $

### 使用sklearn进行xgboost


```python
reg = XGBR(n_estimators=190,random_state=420,eta=0.1
           ,subsample=0.9479166666666666,booster="dart").fit(Xtrain,Ytrain) 
reg.score(Xtest,Ytest) 
```




    0.9261191829728914




```python
MSE(Ytest,reg.predict(Xtest))
```




    6.874897054416443



### 使用xgboost库


```python
import xgboost as xgb
```

##### 使用类DMatrix读取数据


```python
dtrain = xgb.DMatrix(Xtrain,Ytrain) # 特征矩阵和标签都进行一个传入
dtest = xgb.DMatrix(Xtest,Ytest)
```

如果想要查看数据，可以在导入数据进入DMatrix之前在pandas中查看


```python
import pandas as pd
pd.DataFrame(Xtrain)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.03041</td>
      <td>0.0</td>
      <td>5.19</td>
      <td>0.0</td>
      <td>0.515</td>
      <td>5.895</td>
      <td>59.6</td>
      <td>5.6150</td>
      <td>5.0</td>
      <td>224.0</td>
      <td>20.2</td>
      <td>394.81</td>
      <td>10.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.04113</td>
      <td>25.0</td>
      <td>4.86</td>
      <td>0.0</td>
      <td>0.426</td>
      <td>6.727</td>
      <td>33.5</td>
      <td>5.4007</td>
      <td>4.0</td>
      <td>281.0</td>
      <td>19.0</td>
      <td>396.90</td>
      <td>5.29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.23300</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.614</td>
      <td>6.185</td>
      <td>96.7</td>
      <td>2.1705</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>379.70</td>
      <td>18.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.17142</td>
      <td>0.0</td>
      <td>6.91</td>
      <td>0.0</td>
      <td>0.448</td>
      <td>5.682</td>
      <td>33.8</td>
      <td>5.1004</td>
      <td>3.0</td>
      <td>233.0</td>
      <td>17.9</td>
      <td>396.90</td>
      <td>10.21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.05059</td>
      <td>0.0</td>
      <td>4.49</td>
      <td>0.0</td>
      <td>0.449</td>
      <td>6.389</td>
      <td>48.0</td>
      <td>4.7794</td>
      <td>3.0</td>
      <td>247.0</td>
      <td>18.5</td>
      <td>396.90</td>
      <td>9.62</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>349</th>
      <td>0.03871</td>
      <td>52.5</td>
      <td>5.32</td>
      <td>0.0</td>
      <td>0.405</td>
      <td>6.209</td>
      <td>31.3</td>
      <td>7.3172</td>
      <td>6.0</td>
      <td>293.0</td>
      <td>16.6</td>
      <td>396.90</td>
      <td>7.14</td>
    </tr>
    <tr>
      <th>350</th>
      <td>0.12650</td>
      <td>25.0</td>
      <td>5.13</td>
      <td>0.0</td>
      <td>0.453</td>
      <td>6.762</td>
      <td>43.4</td>
      <td>7.9809</td>
      <td>8.0</td>
      <td>284.0</td>
      <td>19.7</td>
      <td>395.58</td>
      <td>9.50</td>
    </tr>
    <tr>
      <th>351</th>
      <td>6.96215</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.700</td>
      <td>5.713</td>
      <td>97.0</td>
      <td>1.9265</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>394.43</td>
      <td>17.11</td>
    </tr>
    <tr>
      <th>352</th>
      <td>0.09164</td>
      <td>0.0</td>
      <td>10.81</td>
      <td>0.0</td>
      <td>0.413</td>
      <td>6.065</td>
      <td>7.8</td>
      <td>5.2873</td>
      <td>4.0</td>
      <td>305.0</td>
      <td>19.2</td>
      <td>390.91</td>
      <td>5.52</td>
    </tr>
    <tr>
      <th>353</th>
      <td>5.58107</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.713</td>
      <td>6.436</td>
      <td>87.9</td>
      <td>2.3158</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>100.19</td>
      <td>16.22</td>
    </tr>
  </tbody>
</table>
<p>354 rows × 13 columns</p>
</div>



#### 写明参数


```python
param = {'objective':'reg:squarederror'
         ,'eta':0.1
         ,'booster':'dart'
         ,'subsample':0.9479166666666666}
num_round = 190 # n_estimators
```

#### 类train，可以直接导入的参数是训练数据，树的数量，其他参数都需要通过params来导入


```python
bst = xgb.train(param, dtrain, num_round)
```

#### 使用接口


```python
preds = bst.predict(dtest)
```

#### 导入sklearn库进行R方和均方误差评估


```python
from sklearn.metrics import r2_score
r2_score(Ytest,preds)
```




    0.9264166709056179




```python
MSE(Ytest,preds)
```




    6.8472146465232635



通过对比发现xgboost库本身是优于sklearn库里xgboost的


```python

```
