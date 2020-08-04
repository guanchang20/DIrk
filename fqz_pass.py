#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import codecs
import pickle
import time
import re
import json
import numpy as np
import pandas as pd
import datetime as dt
import xgboost as xgb
from glob import glob
import statsmodels.api as sm
from scipy.stats import pearsonr
#from pyecharts import Bar, Line, Overlap
#from pyecharts import Page
from os.path import splitext, split, join, exists
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
import os
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


# In[2]:


import sys
sys.path.append('/Users/silver/Desktop/python/qcutils')

from qcutils import qcutils
from analysis_tools import feature_selection as fs
from misc import qctools
from misc import get_order_pass as gop
from misc import get_user_screen as gus
from misc import qnj_prdt_info as gpi
from analysis_tools import data_processing as dp
from analysis_tools import dumb_containers as dc
from analysis_tools import model_monitor as mm
# #from qcutils.drawing_tools import drawing_tools as dtls
from analysis_tools import overdue_ana as oa
from analysis_tools import tools as tls


# In[3]:


from collections import defaultdict
from tqdm import tqdm
import time
import seaborn as sns
from IPython import display
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve


# In[4]:


from sklearn.linear_model import LinearRegression


# In[5]:


from sklearn.ensemble import GradientBoostingClassifier


# ### func

# In[6]:


get_ipython().run_line_magic('pinfo', 'train_test_split')


# In[7]:


def get_train_test_split(data, target):
    """
    样本切分
    """
    
    test_size = 0.3
    x_train, x_test, y_train, y_test = train_test_split(data, data[target],
                                                        test_size=test_size, random_state=2)
    
    x_train = x_train.reset_index(drop = True)
    x_test = x_test.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)
    y_test = y_test.reset_index(drop = True)
    
    return x_train,x_test


# In[8]:


def get_iv_c(x_train, target, cols_c):
    """
    计算cat变量的cwoe
    
    """
     
    ref_table = pd.DataFrame()
    iv_d = dict()
    
    var_no_cwoe = []
    for v in cols_c:
        try:
            if len(x_train[v].unique()) > 500:
                print("---> WARNING: too many unique values for {} to do chi-square merge".format(v))
                var_no_cwoe.extend([v])
                continue
        except KeyError:
    #         print("No column \"{}\" found".format(v))
            continue
        
        if v == target or v == 'CurrentDefaultDays':
            var_no_cwoe.extend([v])
            continue
            
        rt, iv, b_stat = dc.calc_nominal_woe(x_train, v, target)
        if rt is None:
            print("---> WARNING: no reference table returned for {}".format(v))
            continue
        
        iv_d[v] = iv.sum()
        ref_table = pd.concat([ref_table, rt], axis=0, ignore_index=True)
    
    return ref_table, pd.DataFrame.from_dict(iv_d, orient='index').reset_index().rename(columns={'index': 'var', 0: 'iv'}).sort_values('iv', ascending=False)
    


# In[9]:


def get_iv_n(x_train, target, cols_n):
    """
    
    计算num变量的nwoe
    
    """
    tmp = pd.DataFrame(x_train[cols_n].nunique()>1).reset_index()
    woe_numeric_vars = tmp[tmp[0]==True]['index'].tolist()
    
    iv = dict()
    df_numeric_ref_table = pd.DataFrame()
    for var in woe_numeric_vars:
    # for var in ['is_idnum_loc_and_cell_loc_same_city']:
        print('#============ Calculating woe of {} ... ============#'.format(var))
        # ds,ref_table = main_get_ref_table(infile,var)
        try:
            ref_table, b_iv, b_stat = dc.main_get_numeric_ref_table(x_train,var,target,20)
        except KeyError:
            print('---> WARNING: no {} found in data'.format(var))
            woe_numeric_vars.remove(var)
            continue
        iv[var] = [ref_table['IV']]
        # df_ref_table_tmp = pd.DataFrame(ref_table.items(), columns=['Var_Value', 'Ref_Value'])
        df_ref_table_tmp = pd.DataFrame([[r, v] for r, v in ref_table.items()], columns=['Var_Value', 'Ref_Value'])
        df_ref_table_tmp['Var_Name'] = var
        df_numeric_ref_table = pd.concat((df_numeric_ref_table,df_ref_table_tmp),axis = 0)
    
        df_iv = pd.DataFrame.from_dict(iv, orient='index').reset_index()                 .rename(columns={'index': 'var', 0: 'iv'}).sort_values('iv', ascending=False)    
        
    return df_numeric_ref_table, df_iv


# In[10]:


def feas_fillna(x_train, target, cols_n):
    """
    空值填充
    """
    vf = dp.ValueFilling()
    feas_list = []
    na_list = []
    tmp = pd.DataFrame(x_train[cols_n].isnull().sum()>0).reset_index()
    
    for v in tmp[tmp[0]==True]['index'].tolist():
        feas_list.append(v)
        if vf.fit_lin_filling_value(x_train, target, v, vals_to_fill=['nan'])[0][np.nan] in ([np.inf,-np.inf, 'inf', '-inf']):
            na_list.append(x_train[v].unique()[0])
        else:    
            na_list.append(vf.fit_lin_filling_value(x_train, target, v, vals_to_fill=['nan'])[0][np.nan])
    return pd.DataFrame({'feas':feas_list, 'na':na_list})


# In[11]:


def get_information(var_lr, validation_col):
    
    """
    计算psi
    
    """
    
    # get corr
    tmp1 = pd.DataFrame(x_train[var_lr+[target]].corr()[target]).reset_index().rename(columns={'index':'feature','target2':'相关性'})
    
    # get ks auc
    tmp2 = get_each_ks_auc(x_train, var_lr)
    
    # get psi
    tmp4 = pd.DataFrame({'feature':var_lr})
    for v in validation_col:
        df_stats_ref = pd.DataFrame()
        for c in var_lr:
            df_stats_ref = df_stats_ref.append(pd.DataFrame(mm.ModelMonitor.calc_monitor_stats(x_train, eval(v), c)), ignore_index=True)
        df_stats_ref.set_index(['var', 'stats']).unstack()
        idx = df_stats_ref['stats']=='psi'
        xx1 = df_stats_ref[idx][['var','value']].rename(columns={'var':'feature','value':'psi_{}'.format(v)})
        idx = df_stats_ref['stats']=='chisq'
        xx2 = df_stats_ref[idx][['var','value']].rename(columns={'var':'feature','value':'chisq_{}'.format(v)})
        tmp3 = xx1.merge(xx2)
        tmp4 = tmp4.merge(tmp3,on='feature')   


    need = tmp1.merge(tmp2,on='feature')
    need = need.merge(tmp4,on='feature')

    return need


# In[12]:


def get_model_evalute(model, X, y):
    if model == 'sm':
        lm = sm.Logit(y, sm.add_constant(X, prepend=False)).fit(disp=0)
        y_pred = lm.predict(sm.add_constant(X, prepend=False, has_constant='add')) 
    else:
        model.fit(X,y)
        y_pred = model.predict(X)
    acc = roc_auc_score(y, y_pred) 
    
    fpr, tpr, threshold = roc_curve(y, y_pred)
    ks = max(tpr-fpr)
    return lm, acc, ks


def forward_stepwise(model, feas, X, y, thre):    
    for i,f in enumerate(feas):
        if i==0:
            need_feas = []
            need_feas.append(f)
        else:    
            lm, acc_old, ks_old = get_model_evalute(model, X[need_feas], y)
            print('='*10)
            print(need_feas+[f])
            print('='*10)
            lm_new, acc_new, ks_new = get_model_evalute(model, X[need_feas+[f]], y)  
            zs_pval = pd.DataFrame([(v[0], abs(float(v[3])), float(v[4]))
                                for v in lm_new.summary().tables[1].data[1:-1]], columns=['var', 'zscore', 'pval'])
                                
            if zs_pval['pval'].max()>0.05:
                idx = zs_pval['pval']!=zs_pval['pval'].max()
                need_feas_s = list(zs_pval[idx]['var'])
                lm_new, acc_new, ks_new = get_model_evalute(model, X[need_feas_s], y)
                if (ks_new-ks_old) > thre:
                    need_feas = need_feas_s
            else:
                if (ks_new-ks_old) > thre:
                    need_feas = need_feas+[f]

    return need_feas


# In[13]:


def get_each_ks_auc(data, feas):
    auc_dict ={}
    for c in feas:
        try:
            lm = sm.Logit(data[target],sm.add_constant(data[c], prepend=False, has_constant='add')).fit(disp=0) #拟合
            predictions = lm.predict(sm.add_constant(data[c], prepend=False, has_constant='add')) #预测
            fpr, tpr, thresholds = roc_curve(data[target],predictions) #roc_曲线
            roc_auc = auc(fpr, tpr)  # 算出来auc
            auc_dict.update({c:roc_auc})  #dict的添加
        except:
            continue
    auc_df = pd.DataFrame.from_dict(auc_dict,orient='index')
    auc_df = auc_df.rename(columns={0: 'AUC'}).sort_values('AUC',ascending=0)
    auc_df = auc_df.reset_index().rename(columns={'index':'feature'})
    
    ks_dict ={}
    for c in feas:
        try:
            lm = sm.Logit(data[target],sm.add_constant(data[c], prepend=False, has_constant='add')).fit(disp=0) #拟合
            predictions = lm.predict(sm.add_constant(data[c], prepend=False, has_constant='add')) #预测
            fpr, tpr, thresholds = roc_curve(data[target],predictions) #roc_曲线
            ks = max(tpr-fpr)  # 算出来ks
            ks_dict.update({c:ks})  #dict的添加
        except:
            continue
    ks_df = pd.DataFrame.from_dict(ks_dict,orient='index')
    ks_df = ks_df.rename(columns={0: 'ks'}).sort_values('ks',ascending=0)
    ks_df = ks_df.reset_index().rename(columns={'index':'feature'})
    
    auc_df = auc_df.merge(ks_df,on = 'feature',how = 'left')
    return auc_df


# In[14]:


def draw_learning_curves(X, y, estimator, num_trainings):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=10, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.plot(test_scores_mean, 'o-', color="y",
             label="Cross-validation score")
    
    plt.plot(train_scores_mean, 'o-', color="g",
             label="Training score")

    plt.legend(loc="best")

    plt.show()


# In[15]:


def evaluate_model_cv(feas, model, nv_num, X, y, thred):   
    score_fold_train_, score_fold_test_, ks_train_, ks_test_ = [],[],[],[] 
    kfold = StratifiedKFold(n_splits=nv_num, random_state=2019, shuffle=True)
    col=[]
    
    for index, (train_index, test_index) in enumerate(kfold.split(X, y)):
        trainx = X.iloc[train_index].reset_index(drop=True)
        trainy = y.iloc[train_index].reset_index(drop=True)
        testx = X.iloc[test_index].reset_index(drop=True)
        testy = y.iloc[test_index].reset_index(drop=True)    
        
        if model == 'sm':
            lm = sm.Logit(trainy, sm.add_constant(trainx, prepend=False)).fit(disp=0)
            trainy_pred = lm.predict(sm.add_constant(trainx, prepend=False, has_constant='add'))         
            testy_pred = lm.predict(sm.add_constant(testx, prepend=False, has_constant='add'))
            cols = forward_stepwise('sm', feas, trainx, trainy, thred)
            col = col+cols
        else:
            model.fit(trainx, trainy)
            trainy_pred = model.predict(trainx)
            testy_pred = model.predict(testx)
        
        
        # evaluate auc
        score_fold_train = roc_auc_score(trainy, trainy_pred)    
        score_fold_test = roc_auc_score(testy, testy_pred)    
        score_fold_train_.append(score_fold_train)
        score_fold_test_.append(score_fold_test)
        
        # evaluate ks
        fpr_train, tpr_train, threshold_train = roc_curve(trainy, trainy_pred)
        ks_train = max(tpr_train-fpr_train)  
        ks_train_.append(ks_train)  
        fpr_test, tpr_test, threshold_test = roc_curve(testy, testy_pred)
        ks_test = max(tpr_test-fpr_test)  
        ks_test_.append(ks_test)   
        print('item is {}'.format(index))
        print('train ks is {}, auc is {}'.format(ks_train, score_fold_train))      
        print('test ks is {}, auc is {}'.format(ks_test, score_fold_test))
        print(cols)
        print('='*10)
        

    print('train_acc mean is {}, train_ks mean is {}'.          format(np.array([score_fold_train_]).mean(), np.array([ks_train_]).mean()))
    print('test_acc mean is {}, test_ks mean is {}'.          format(np.array([score_fold_test_]).mean(), np.array([ks_test_]).mean()))
    
    print(list(set(col)))
    
    return np.array([score_fold_train_]).mean()


# In[16]:


def get_information(var_lr, validation_col):
    # get corr
    tmp1 = pd.DataFrame(x_train[var_lr+[target]].corr()[target]).reset_index().rename(columns={'index':'feature','target2':'相关性'})
    
    # get ks auc
    tmp2 = get_each_ks_auc(x_train, var_lr)
    
    # get psi
    tmp4 = pd.DataFrame({'feature':var_lr})
    for v in validation_col:
        df_stats_ref = pd.DataFrame()
        for c in var_lr:
            df_stats_ref = df_stats_ref.append(pd.DataFrame(mm.ModelMonitor.calc_monitor_stats(x_train, eval(v), c)), ignore_index=True)
        df_stats_ref.set_index(['var', 'stats']).unstack()
        idx = df_stats_ref['stats']=='psi'
        xx1 = df_stats_ref[idx][['var','value']].rename(columns={'var':'feature','value':'psi_{}'.format(v)})
        idx = df_stats_ref['stats']=='chisq'
        xx2 = df_stats_ref[idx][['var','value']].rename(columns={'var':'feature','value':'chisq_{}'.format(v)})
        tmp3 = xx1.merge(xx2)
        tmp4 = tmp4.merge(tmp3,on='feature')   


    need = tmp1.merge(tmp2,on='feature')
    need = need.merge(tmp4,on='feature')

    return need


# In[17]:


def get_auc_ks(x_train,x_test,var_lr):
    auc_ks = get_each_ks_auc(x_train, var_lr)
    auc_ks = auc_ks.rename(columns={'AUC':'AUC_train', 'ks':'ks_train'})
    
    auc_ks1 = get_each_ks_auc(x_test, var_lr)
    auc_ks1 = auc_ks1.rename(columns={'AUC':'AUC_test', 'ks':'ks_test'})   
    
    auc_ks = auc_ks.merge(auc_ks1)
    return auc_ks


# In[ ]:





# ### read_data

# In[127]:


target = 'FPD_15'


# In[128]:


train = pd.read_csv(r'/Users/silver/Desktop/data/reject_inference/gbdt_esmm/_pass_pre.train.csv')
test = pd.read_csv(r'/Users/silver/Desktop/data/reject_inference/gbdt_esmm/_pass_pre.test.csv')

print(train.shape)
train.head(2)


# In[129]:


train.PAYDATE.min(),train.PAYDATE.max()


# In[130]:


asd = pd.DataFrame(data3.count()/len(data3), columns=['rto'])
train.drop(list(asd.loc[asd['rto']<0.3].index), axis=1, inplace=True)


# In[131]:


ob_cols = train.columns[6:-7]


# In[132]:


ob_cols


# In[133]:


train[target]


# In[134]:


base_col = ["PAYDATE"]


# In[135]:


del_col = ["FPD_7"]


# In[136]:


train = train[list((set(list(train.columns)))-set(list(train[del_col]))-set(list(train[base_col])))]


# In[137]:


test = test[list((set(list(test.columns)))-set(list(test[del_col]))-set(list(test[base_col])))]


# In[138]:


train.FPD_15.value_counts()


# ### split

# In[139]:


train


# ### 计算iv值

# #### cwoe

# In[140]:


cols_c = train.select_dtypes(include='object').columns   
print(len(cols_c))
cols_c[:2]


# In[92]:


cwoe_table, iv_c = get_iv_c(train, target, cols_c)


# In[141]:


train


# In[142]:


dc.set_nominal_woe(train, cwoe_table)
dc.set_nominal_woe(test, cwoe_table)
print(train.shape, test.shape)


# #### fillna

# In[143]:


cols_n = list(set(train.select_dtypes(include=np.number).columns) - {c for c in train if re.search('cwoe$',c)} - {target})
print(len(cols_n))
cols_n[:5]


# In[144]:


val_fill_cfg = feas_fillna(train, target, cols_n)


# In[145]:


for f in val_fill_cfg['feas']:
    train[f] = train[f].fillna(val_fill_cfg[val_fill_cfg['feas']==f]['na'].tolist()[0])
    test[f] = test[f].fillna(val_fill_cfg[val_fill_cfg['feas']==f]['na'].tolist()[0])


# #### nwoe

# In[146]:


nwoe_table, iv_n = get_iv_n(train, target, cols_n)


# In[147]:


iv = pd.concat([iv_c,iv_n],axis=0)
print(iv.shape)
iv.head()


# ### 算ks_auc

# In[148]:


iv = iv.sort_values('iv', ascending=False).reset_index(drop=True)
print(iv.shape)
iv.head(2)


# In[149]:


var_lr = iv[iv['iv']>0.01]['var'].tolist()
print(len(var_lr))
var_lr[:2]


# In[150]:


var_lr


# In[151]:


get_auc_ks(train,test,var_lr).merge(get_information(var_lr, ['x_test'])[['feature', target, 'psi_x_test']].                                        rename(columns={target:'相关性'}))


# In[152]:


get_auc_ks(train,test,var_lr).merge(get_information(var_lr, ['x_test'])[['feature', target, 'psi_x_test']].                                        rename(columns={target:'相关性'})).to_csv('/Users/silver/Desktop/data/reject_inference/gbdt_esmm/result/auc_ks_psi_corr.csv')


# In[153]:


var_lr = get_auc_ks(train,test,var_lr).merge(get_information(var_lr, ['x_test'])[['feature', target, 'psi_x_test']].                                        rename(columns={target:'相关性'}))['feature'].tolist()
print(len(var_lr))
var_lr[:10]


# In[118]:


var_lr


# In[154]:


cols = forward_stepwise('sm', var_lr, train[var_lr], train[target], 0.002)
print(len(cols))
print(cols)


# In[155]:


train[cols].corr()


# In[156]:


v = train
lm = sm.Logit(v[target], sm.add_constant(v[cols], prepend=False)).fit(disp=0)
_ = tls.lr_model(v[cols].values,v[target], lm, rtn_pred=False)
plt.show()
lm.summary()


# In[157]:


v = test
_ = tls.lr_model(v[cols].values,v[target].values, lm, rtn_pred=False)
plt.show()


# In[124]:


var_ob = ['TD_TONGDUN_ANTI_SCORE',
 'BR_JD_M3_ID_NBK_ALLNUM',
 'BR_JD_M6_ID_NBK_ALLNUM',
 'BR_JD_M3_CL_NBK_ALLNUM',
 'BR_JD_M6_CL_NBK_ALLNUM',
 'BR_JD_M3_AVG_MONNUM',
 'JZ_AIR_TRAV_REP_L36M_TRAV_NUM',
 'BR_JD_M3_ID_CAON_ALLNUM',
 'BR_TRAINCCZCS_TRAIN_NUM',
 'BR_JD_M1_ID_NBK_ALLNUM',
 'BR_JD_M1_NBK_ORGNUM',
 'BR_JD_M3_CL_CAON_ALLNUM',
 'SC_L12M_QUERY_NUM',
 'BR_JD_M12_ID_NBK_ALLNUM',
 'TX_ANTIFRDVIP_RISKSCORE',
 'BR_JD_M1_CL_NBK_ALLNUM',
 'BR_JD_D15_ID_NBK_ALLNUM',
 'BR_JD_M12_CL_CAOFF_ORGNUM',
 'BR_JD_M12_CL_NBK_ALLNUM',
 'BR_JD_M12_ID_CAOFF_ORGNUM',
 'BR_JD_D15_CL_NBK_ALLNUM',
 'AC_BR_PH_ONL_TM_PH_ONL_SEC',
 'BR_JD_M6_ID_NBKNSLN_ORGNUM',
 'BD_MULTI_COMPANYADDR_CUST_NUM',
 'BR_JD_M3_ID_COOFF_ALLNUM',
 'LK_AGE',
 'BR_JD_M12_ID_NBK_WK_ORGNUM',
 'BR_JD_M12_ID_NBKSLN_ORGNUM',
 'BR_JD_M6_CL_NBKNSLN_ORGNUM',
 'BR_JD_M12_CL_NBKSLN_ORGNUM',
 'BD_MULTI_CONM_CUST_NUM',
 'BR_JD_M12_ID_NBK_NT_ALLNUM',
 'BR_JD_M12_CL_NBK_WK_ORGNUM',
 'BR_JD_M3_CL_COOFF_ALLNUM',
 'BR_JD_M12_CL_NBK_NT_ALLNUM',
 'SC_TRAINCFXFZJE_TOTAL_CONS_AMT',
 'YZ_ONERELNUM',
 'YZ_REFIDNOCNTL3M',
 'YL_L12M_TRANS_COUNT',
 'LK_APPLY_HOUR',
 'BR_JD_M6_ID_NBK_NT_ALLNUM',
 'BR_JD_M6_CL_NBK_NT_ALLNUM',
 'YZ_USEDIFFIDNOCNT',
 'LK_IdNo_1',
 'YZ_FRAUDSCORE',
 'YL_IS_HIGH_CUSTOMER',
 'BR_JD_M3_ID_NBK_NT_ALLNUM',
 'BR_JD_M3_CL_NBK_NT_ALLNUM',
 'BD_MULTI_CELL_F7DIGI_NUMBER']


# In[158]:


train[cols]


# In[159]:


train[target]


# In[275]:


x_train_tmp1 = x_train[var_ob].merge(x_train[target],left_index=True,right_index=True)
x_test_tmp1 = x_test[var_ob].merge(x_test[target],left_index=True,right_index=True)


# In[280]:


x_train_tmp1


# ### 样本均衡

# In[291]:


label1 = x_train_tmp1.ix[x_train_tmp1["FPD_15"]>0,]


# In[292]:


label1


# In[293]:


i =20
data_label = label1
while i>1:
    data_label=pd.concat([data_label,label1],axis=0).reset_index(drop=True)
    i=i-1


# In[297]:


label0 = x_train_tmp1.ix[x_train_tmp1["FPD_15"]<1,]


# In[299]:


data_train_balance = pd.concat([data_label,label0],axis=0).reset_index(drop=True)


# In[301]:


data_train_balance.shape


# In[294]:


data_label["FPD_15"]


# In[230]:


x_train_tmp1


# In[216]:


x_train_tmp.FPD_15.value_counts()


# In[217]:


1769/(35916+1769)


# In[218]:


x_test_tmp.FPD_15.value_counts()


# In[220]:


768/(768+15383)


# In[257]:


var1_train = x_train[["TD_TONGDUN_ANTI_SCORE"]].merge(x_train[target],left_index=True,right_index=True)
var1_test = x_test[["TD_TONGDUN_ANTI_SCORE"]].merge(x_test[target],left_index=True,right_index=True)


# In[258]:


var1_train


# In[231]:


x_test_tmp1 = x_test[var_ob].merge(x_test[target],left_index=True,right_index=True)


# In[276]:


x_train_tmp1.to_csv(r"D:\jupyter_ctj\data\lr_0706\x_train_tmp1.csv")
x_test_tmp1.to_csv(r"D:\jupyter_ctj\data\lr_0706\x_test_tmp1.csv")


# ### score

# In[199]:


x_train["score"] = 1 / (1 + np.exp(-1 * (x_train["ft_lbs_pwoi_all_often_consum_middle"]*63.1889
                                          +x_train['ft_safe_score_e_rule']*0.0147
                                         -x_train['ft_app_ins_current_travel_cnt_avg']*0.0907
                                         -x_train['ft_lbs_same_night_stay_wifimac_cnt']*0.0456
                                         +x_train['ft_lbs_home_cons_ls_low']*3.5174
                                         -x_train['ft_dev_deprecia_price']*0.2367
                                         -x_train['ft_stable_imsi_changes']*0.3939
                                         -42.1120)))


# ### 训练GBDT

# In[232]:


x_train_tmp[var_ob]


# In[262]:


var1 = ["TD_TONGDUN_ANTI_SCORE"]


# In[ ]:





# In[303]:


data_train_balance[var_ob]


# In[304]:


X_train=data_train_balance[var_ob]
X_test=x_test_tmp1[var_ob]
y_train=data_train_balance["FPD_15"]
y_test=x_test_tmp1["FPD_15"]


# In[305]:


X_train.info()


# In[306]:


X_train[var_lr]


# In[335]:


gbm1 = GradientBoostingClassifier(n_estimators=50, random_state=10, subsample=0.5, max_depth=2,learning_rate=0.1)


# In[336]:


gbm1.fit(X_train, y_train)


# In[337]:


predictions_gbdt = gbm1.predict(X_train[var_ob])
actuals_gbdt = y_train

dc.evaluate_performance(actuals_gbdt.values, predictions_gbdt)


# In[338]:


predictions_gbdt = gbm1.predict(X_test[var_ob])
actuals_gbdt = y_test

dc.evaluate_performance(actuals_gbdt.values, predictions_gbdt)


# In[370]:


x_train


# In[378]:


col_0707 = x_train.iloc[:,0:119].columns


# In[ ]:


x_train.iloc[:,0:52]


# #### 特征组合（树）

# In[362]:


train_new_feature_gbdt= gbm1.apply(X_train)


# In[364]:


train_gbdt = pd.DataFrame(train_new_feature_gbdt.reshape(-1, 50))


# In[372]:


train_gbdt


# #### 组合特征验证

# In[385]:


train_gbdt[0].value_counts()


# In[ ]:





# In[393]:


idx1 = X_train['TD_TONGDUN_ANTI_SCORE']<=59.5
idx2 = X_train['TX_ANTIFRDVIP_RISKSCORE']<=4.5
idx3 = X_train['BD_MULTI_CONM_CUST_NUM']<=0.5

print(X_train[idx1&idx2].shape[0])   #2
print(X_train[idx1&~idx2].shape[0])  #3
print(X_train[~idx1&idx3].shape[0])  #5
print(X_train[~idx1&~idx3].shape[0]) #6


# In[391]:





# In[387]:


trees = gbm1.estimators_.ravel()
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
def plot_tree(gbm1):
    dot_data = StringIO()
    export_graphviz(gbm1, out_file=dot_data, node_ids=True,feature_names=var_lr,
                    filled=True, rounded=True, 
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())

# now we can plot the first tree
plot_tree(trees[0])


# In[366]:


test_new_feature_gbdt = gbm1.apply(X_test)


# In[367]:


test_gbdt = pd.DataFrame(test_new_feature_gbdt.reshape(-1, 50))


# In[ ]:


train_gbdt01 = x_train.iloc[:,0:124].merge(train_gbdt,left_index=True, right_index=True)
test_gbdt01 = x_test.iloc[:,0:124].merge(test_gbdt,left_index=True, right_index=True)


# In[ ]:


test_gbdt01.head()


# #### 组合特征(节点)

# ##### train

# In[401]:


x_train_gbdt0506 = train_gbdt.copy()


# In[402]:


#### 新特征数据
col_train_gbdt0506 = list(train_gbdt.columns)


#### 还原原数据
x_train01_gbdt_f52 = X_train.iloc[:,0:49]
#x_train_gbdt0506 = X_train[col_train_gbdt0506].astype('str')


#### 批量改列名

prefix_columns_train_gbdt = []
for column_name in x_train_gbdt0506.columns:
    if column_name not in ["index", 'FPD_15']:
        prefix_columns_train_gbdt.append('gbdt_fea' + '_' + str(column_name))
    else:
        prefix_columns_train_gbdt.append(column_name)     
x_train_gbdt0506.columns = prefix_columns_train_gbdt 



x_train_gbdt0506_copy = x_train_gbdt0506.copy()


#### one_hot处理
x_train_gbdt0506_copy[prefix_columns_train_gbdt] = x_train_gbdt0506_copy[prefix_columns_train_gbdt].astype('str')
pro_fea_fin2_train_gbdt = pd.get_dummies(x_train_gbdt0506_copy[prefix_columns_train_gbdt])




# In[408]:


import math


# In[409]:


class WOE:
    def __init__(self):
        self._WOE_MIN = -5
        self._WOE_MAX = 5

    def count_binary(self, a, event=1):
        event_count = (a == event).sum()
        non_event_count = a.shape[-1] - event_count
        return event_count, non_event_count
        
    def woe_single_x(self, x, y, event=1):
        event_total, non_event_total = self.count_binary(y, event=event)
        # x_labels = np.unique(x)
        x_labels = pd.Series(x).unique()
        woe_dict = {}
        iv = 0
        for x1 in x_labels:
            y1 = y.iloc[np.where(x == x1)[0]]
            event_count, non_event_count = self.count_binary(y1, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe1 = self._WOE_MIN
            elif rate_non_event == 0:
                woe1 = self._WOE_MAX
            else:
                woe1 = math.log(rate_event / rate_non_event)
            woe_dict[x1] = woe1
            iv += (rate_event - rate_non_event) * woe1
        return woe_dict, iv


# In[410]:


def iv_selection(df,cols,target):

    feas = []
    iv = []
    func = WOE()
    df_tmp = df.copy()
    for f in tqdm(cols):
        feas.append(f)
        if df_tmp[f].nunique()>20:
            bins = list(np.linspace(df_tmp[f].min(),df_tmp[f].max(),20))
            df_tmp[f+'_bin'] = pd.cut(df_tmp[f], bins)
            df_tmp[f+'_bin'] = df_tmp[f+'_bin'].apply(lambda x:get_range(str(x)))
            df_tmp[f+'_bin'] = df_tmp[f+'_bin'].astype(float)
            df_tmp[f+'_bin'] = df_tmp[f+'_bin'].fillna(-1) 
            iv.append(func.woe_single_x(df_tmp[f+'_bin'],df_tmp[target])[1])
        else:
            print(f)
            df_tmp[f] = df_tmp[f].fillna(-1) 
            iv.append(func.woe_single_x(df_tmp[f],df_tmp[target])[1])
    
    iv_feas = pd.DataFrame(iv,feas).reset_index().rename(columns={'index':'feas',0:'iv'})
    iv_feas = iv_feas.sort_values('iv',ascending=False).reset_index(drop=True)
    return iv_feas


# #### 特征筛选

# In[416]:


data_rank_train_gbdt_copy.iloc[:,0:200]


# In[427]:


#### 和y 拼接(stop if data is test/oot)
data_rank_train_gbdt  = pro_fea_fin2_train_gbdt.merge(data_train_balance["FPD_15"],left_index=True, right_index=True)


#### 特征筛选
data_rank_train_gbdt_copy =data_rank_train_gbdt.copy()
cols_gbdt_nf_fin = list(data_rank_train_gbdt_copy.iloc[:,0:200].columns)
target = "FPD_15"

df_train_gbdt = iv_selection(data_rank_train_gbdt_copy,cols_gbdt_nf_fin,target)

col_fin_gbdt_train_0506 = list(df_train_gbdt.head(50)["feas"])
onehot_df_train_gbdt = data_rank_train_gbdt[col_fin_gbdt_train_0506].merge(data_train_balance[[target]],left_index=True, right_index=True)


# In[414]:


onehot_df_train_gbdt


# In[422]:





# In[419]:


data_train_balance


# In[428]:


v = onehot_df_train_gbdt
cols_v = onehot_df_train_gbdt.iloc[:,:50].columns
lm = sm.Logit(v[target], sm.add_constant(v[cols_v], prepend=False)).fit(disp=0)
_ = tls.lr_model(v[cols_v].values,v[target], lm, rtn_pred=False)
plt.show()
lm.summary()


# In[434]:


cols_fin = forward_stepwise('sm', cols_v, v[cols_v], v[target], 0.002)
print(len(cols_fin))
print(cols_fin)


# In[435]:


v = onehot_df_train_gbdt
#cols_v = onehot_df_train_gbdt.iloc[:,:25].columns
lm = sm.Logit(v[target], sm.add_constant(v[cols_fin], prepend=False)).fit(disp=0)
_ = tls.lr_model(v[cols_fin].values,v[target], lm, rtn_pred=False)
plt.show()
lm.summary()


# In[ ]:





# In[ ]:





# ##### test

# In[437]:


#### 新特征数据
col_test_gbdt = test_gbdt.merge(x_test["FPD_15"],left_index=True, right_index=True)


#### 还原原数据
x_test01_xgb_f52 = x_test.iloc[:,0:49]
x_test_gbdt = x_test[col_test_gbdt].astype('str')


#### 批量改列名

prefix_columns_test_gbdt = []
for column_name in test_gbdt.columns:
    if column_name not in ["index", 'FPD_15']:
        prefix_columns_test_gbdt.append('gbdt_fea' + '_' + str(column_name))
    else:
        prefix_columns_test_xgb.append(column_name)     
x_test_gbdt.columns = prefix_columns_test_xgb 



x_test_gbdt_copy = x_test_gbdt.copy()


#### one_hot处理
x_test_gbdt_copy[prefix_columns_test_xgb] = x_test_gbdt_copy[prefix_columns_test_xgb].astype('str')
pro_fea_fin2_test_xgb = pd.get_dummies(x_test_gbdt_copy[prefix_columns_test_xgb])



#### 和y 拼接(stop if data is test/oot)
data_fin_test_xgb  =  pro_fea_fin2_test_xgb[col_fin_gbdt_train_0506].merge(x_test[[target]],left_index=True, right_index=True)


# ### 训练XGB

# In[339]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import logging
import xgboost as xgb
import time
from sklearn.datasets import load_iris

logging.basicConfig(format='%(asctime)s : %(levelname)s: %(message)s', level=logging.INFO)


# In[340]:


params = {
            'max_depth': 2,
            'learning_rate': 0.07,
            'min_child_weight': 5,
            'max_delta_step': 0.7,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'silent': True,
            'objective': 'binary:logistic',
            'missing': None,
            'eval_metric': 'auc',
            'seed': 2020,
            'n_estimators':50,
            'booster':'dart'
    
}


# In[341]:


model = xgb.XGBClassifier(**params)


# In[ ]:


X_train=data_train_balance[var_ob]
X_test=x_test_tmp1[var_ob]
y_train=data_train_balance["FPD_15"]
y_test=x_test_tmp1["FPD_15"]


# In[342]:


model.fit(X_train, y_train)


# In[343]:


predictions = model.predict(X_train)
actuals = y_train

dc.evaluate_performance(actuals.values, predictions)


# In[344]:


predictions = model.predict(X_test)
actuals = y_test

dc.evaluate_performance(actuals.values, predictions)


# ### 训练rf

# In[358]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn import tree
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Random Forests in `scikit-learn` (with N = 100)
rf = RandomForestClassifier(n_estimators=65,max_depth=2,criterion='entropy',
                            random_state=10)
rf.fit(X_train, y_train)


# In[359]:


train_rf = rf.apply(X_train)
test_rf = rf.apply(X_test)


# In[360]:


predictions_rf = rf.predict(X_train)
actuals_rf = y_train

dc.evaluate_performance(actuals_rf.values, predictions_rf)


# In[361]:


predictions_rfy = model.predict(X_test)
actuals_rfy = y_test

dc.evaluate_performance(actuals_rfy.values, predictions_rfy)


# In[ ]:





# In[ ]:





