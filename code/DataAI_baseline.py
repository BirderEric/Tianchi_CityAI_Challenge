import pandas as pd
import copy
import logging
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error
import warnings
import lightgbm as lgb
import xgboost as xgb
import os
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import warnings
import matplotlib.pyplot as plt
import copy
import numpy as np
import pickle
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
ori_feature = ['time','lineID',	'stationID','deviceID','status','userID','payType']

def load_file(daystart,dayend):
    df = pd.DataFrame()
    len_sheet = []
    for i in range(int(daystart),int(dayend)+1):
        if i < 10:
            sheet =  pd.read_csv('../data/train/record_2019-01-0' + str(i) + '.csv')
        else:
            sheet =  pd.read_csv('../data/train/record_2019-01-' + str(i) + '.csv')
        len_sheet.append(len(sheet))
        df = pd.concat([df,sheet])
    return df,len_sheet
def time_pro(sheet1):
    
    sheet1['time'] = sheet1['time'].apply(lambda x:str(x))
    #+ datetime.timedelta(minutes=8)
    sheet1['month'] = sheet1['time'].apply(lambda x:int(x[5:7]))
    sheet1['day'] = sheet1['time'].apply(lambda x:int(x[8:10]))
    sheet1['hour'] = sheet1['time'].apply(lambda x:int(x[11:13]))
    sheet1['min'] = sheet1['time'].apply(lambda x:int(x[14:16] ))
    return sheet1        
import copy

def perSt_perDay_perTen_count(df):
    clip = [0]
    Day_sheet = pd.DataFrame()
    Day_sheet['Standard_time'] = [t for t in range(24*6)]
    
    for _clip in clip:
        df['map_time'] = df.hour * 6 + df['time_clip_' + str(_clip)]       
        Df = copy.deepcopy(df)
        for St in df.stationID.unique():
            Day_sheet['Station_' + str(St)] = St
            df1 = Df[Df['stationID'] == St]
            temp = df1.groupby(['map_time','status'],as_index=False)['userID'].count()
            df_in = temp[temp['status'] == 1]
            df_out = temp[temp['status'] == 0]
            dict_in = dict(zip(df_in['map_time'].values,df_in['userID'].values))
            dict_out = dict(zip(df_out['map_time'].values,df_out['userID'].values))
            Day_sheet['In_' + str(St) + '_' + str(_clip)] = Day_sheet['Standard_time'].apply(lambda x:dict_in[x] if x in dict_in.keys() else 0)
            Day_sheet['out_' + str(St) + '_' + str(_clip)] = Day_sheet['Standard_time'].apply(lambda x:dict_out[x] if x in dict_out.keys() else 0)
    return Day_sheet
def time_clip(df):
    clip = [0]
    for _clip in clip:
        df['time_clip_' + str(_clip)] = df['min'].apply(lambda x:int((x+_clip)/10))
    return df
def findpeak(array):
    index = []
    for i in range(1,len(array)-1):
        if array[i] > array[i-1] and array[i] > array[i+1]:
            index.append(i)
    return index
def findvalley(array):
    index = []
    for i in range(1,len(array)-1):
        if array[i] < array[i-1] and array[i] < array[i+1]:
            index.append(i)
    return index

def day_res_extract(df,daystart,dayend):
    res = pd.DataFrame()
    for i in range(daystart,dayend+1):
        res_ = perSt_perDay_perTen_count(df.loc[df['day'] == i])
        res_['day'] = i
        res = pd.concat([res,res_],axis=0)
    return res

def gen_train_data(df,label_day,stationID,flag):
    
    data = pd.DataFrame()
    if flag:
        label = df.loc[df['day'] == label_day]
    time_ = []
    for station in stationID:
        cnt = 0
        for i in range(1,6):
            if label_day-i in df.day.unique():
                time_.append(i)
                df_ = df.loc[df['day'] == label_day-i]
                cnt += 1
                if cnt == 1:
                    temp = df_[['Standard_time',
                               'Station_'+str(station),
                               'In_'+str(station)+'_0',
                               'out_'+str(station)+'_0']]
                else:
                    temp = pd.concat([temp,df_[['In_' + str(station) + '_0',
                                               'out_' + str(station) + '_0']]],axis=1)
            elif cnt == 5:
                break
        temp.columns = ['Standard_time', 'sta', 'In1', 'out1', 'In2',
                        'out2', 'In3', 'out3', 'In4', 'out4','In5', 'out5']
        if flag:
            temp['labelin'] = label['In_' + str(station) + '_0'].values
            temp['labelout'] = label['out_' + str(station) + '_0'].values
        else:
            temp['labelin'] = 0
            temp['labelout'] = 0
        data = pd.concat([data,temp],axis=0)
    data['day'] = label_day
    return data,time_[:5]
      
def shift(df):
    for i in range(1,6):
        df['In_'+str(i)+'_1'] = df['In'+str(i)].shift(1)
        df['In_'+str(i)+'_-1'] = df['In'+str(i)].shift(-1)
        df['out_'+str(i)+'_1'] = df['out'+str(i)].shift(1)
        df['out_'+str(i)+'_-1'] = df['out'+str(i)].shift(-1)
    return df    

def load_train_file():
    tra1_13,len1_13 = load_file(1,13)
    tra14_25,len14_25 = load_file(14,25)
    tra1_13 = time_pro(tra1_13)
    tra1_13 = time_clip(tra1_13)
    tra14_25 = time_pro(tra14_25)
    tra14_25 = time_clip(tra14_25)
    tra1_13 = tra1_13.drop(['time','lineID'],axis=1)
    tra14_25 = tra14_25.drop(['time','lineID'],axis=1)
    return tra1_13,tra14_25
def gen_train_valid(df1_13,df14_25):
    
    train_1_13 = copy.deepcopy(df1_13)
    train_14_25 = copy.deepcopy(df14_25)
    all_res_1 = day_res_extract(train_1_13,1,13)
    all_res = day_res_extract(train_14_25,14,25)
    all_res_1_25 = pd.concat([all_res,all_res_1],axis=0)
    test = pd.read_csv('../data/testB/testB_record_2019-01-26.csv')
    test = time_pro(test)
    test = time_clip(test)
    test = test.drop(['time','lineID'],axis=1)
    valid = day_res_extract(test,26,26)
    train_data = pd.DataFrame()
    res1_26 = pd.concat([all_res_1_25,valid],axis=0)
    train_data = pd.DataFrame()
    for i in [6,13,20,27]:
        if i == 27 :
            a,b = gen_train_data(res1_26,i,train_1_13.stationID.unique(),0)
        else:
            a,b = gen_train_data(res1_26,i,train_1_13.stationID.unique(),1)
        for idx,v in enumerate(b):
            a[str(idx)] = v
        train_data = pd.concat([train_data,a],axis=0)
    return train_data

def train(df,trainlabel='labelin',ifvalid=True):

    train_data_= copy.deepcopy(df)
    if ifvalid:
        train_ = copy.deepcopy(train_data_.loc[train_data_['day'] < 20 ].reset_index())
        valid_28 = copy.deepcopy(train_data_.loc[train_data_['day'] == 20 ].reset_index())
        train_ = shift(train_)
        valid_28 = shift(valid_28)
    else:
        train_ = copy.deepcopy(train_data_.loc[train_data_['day'] < 27 ].reset_index())
        valid_28 = copy.deepcopy(train_data_.loc[train_data_['day'] == 27 ].reset_index())
        train_ = shift(train_)
        valid_28 = shift(valid_28)
    pred = ['Standard_time', 'sta', 'In1', 'In2', 'In3',
        'In4', 'In5' , '0', '1', '2', '3',
        '4']
    pred = [i for i in train_.columns if i not in ['labelin','labelout','index','sum', 'mean', 'max']]
    n_splits = 5
    oof_lgb = np.zeros(len(train_))
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=2019)
    sub_preds = np.zeros(valid_28.shape[0])
    res_e = []
    use_cart = True
    cate_cols = ['sta']
    label = trainlabel
    params = {
        'learning_rate': 0.02,
        'boosting_type': 'gbdt',
        'objective':'regression',
        'metric': 'mae',
        'num_leaves': 30,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 11,
        'min_data_in_leaf': 20,
        'max_depth':5,
        'nthread': -1,
        'verbose': -1,
        #'lambda_l2':0.7
    }
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_[pred]), start=1):
        print('the %s training start ...'%n_fold)
        train_x, train_y = train_[pred].iloc[train_idx], train_[label].iloc[train_idx]
        valid_x, valid_y = train_[pred].iloc[valid_idx], train_[label].iloc[valid_idx]

        if use_cart:
            dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=cate_cols)
            dvalid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=cate_cols)
        else:
            dtrain = lgb.Dataset(train_x, label= train_y)
            dvalid = lgb.Dataset(valid_x, label= valid_y)

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=20000,
            valid_sets=[dvalid],
            early_stopping_rounds = 100,
            verbose_eval=200
        )
        sub_preds += clf.predict(valid_28[pred],num_iteration=clf.best_iteration)/ folds.n_splits
        train_pred = clf.predict(valid_x,num_iteration=clf.best_iteration)
        oof_lgb[valid_idx] = train_pred
        print('*' * 100)
    return sub_preds

def cal_mae(pre,label):

    print(mean_absolute_error(pre,label))

def gen_submit(df1,df2):
  
    train_data = gen_train_valid(df1,df2)
    sub_prein = train(df=train_data,trainlabel='labelin',ifvalid=True)
    sub_preout = train(df=train_data,trainlabel='labelout',ifvalid=True)
    submit = pd.read_csv('../data/testB/testB_submit_2019-01-27.csv')
    test = copy.deepcopy(train_data.loc[train_data['day'] == 20 ].reset_index())
    
    test['inNums'] = sub_prein
    test['inNums'] = test['inNums'].apply(lambda x : int(x))
    test['outNums'] = sub_preout
    test['outNums'] = test['outNums'].apply(lambda x : int(x))
    #my_best_sub = pd.read_csv('my_second_model_int.csv')
    for sta in submit.stationID.unique():
        if sta == 54:
            submit.loc[submit['stationID'] == sta,'inNums'] = 0
            submit.loc[submit['stationID'] == sta,'outNums'] = 0
        else:
            submit.loc[submit['stationID'] == sta,'inNums'] = test.loc[test['sta'] == sta]['inNums'].values
            submit.loc[submit['stationID'] == sta,'outNums'] = test.loc[test['sta'] == sta]['outNums'].values
    return submit

def main():

    tra1_13,tra14_25 = load_train_file()
    submit_B = gen_submit(tra1_13,tra14_25)
    submit_B.to_csv('../submit/testB/DataAIbaseline.csv',index=False)

if __name__ == "__main__":  
    main()
