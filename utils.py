import pandas as pd
import numpy as np
import pickle
import scikitplot as skplt
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,roc_curve, auc, roc_auc_score, precision_recall_curve,average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def join_split(df1,df2,df3,df4,tag):
    """This function gets the data jions it and then split it in x and y
    df1-4: are the dataframes that will be joined
    tag= target column in the data

    return:
    X= The feature columns
    y= the targer column
    
    """
    data=pd.concat([df1,df2,df3,df4])
    target=tag
    numeric_var=list(data.loc[:, data.columns != tag].columns)
    X=data[numeric_var]
    y=data[target]
    return(data,X,y)

def split_data(df,tag):
    """This function gets the data then split it in x and y
    df: is the dataframe that will be joined
    tag= target column in the data

    return:
    X= The feature columns
    y= the targer column
    
    """
    data=df
    target=tag
    numeric_var=list(data.loc[:, data.columns != tag].columns)
    X=data[numeric_var]
    y=data[target]
    return(X,y)

selected_features=['Ionnqty_w1', 'callqty_w1', 'Ionndur_w1', 'callqty_w2',
       'MIcalldur_ch1', 'm1actdcon', 'onnet_all', 'call_qty_all',
       'm1diffchrgtr', 'MSISDN', 'rev_w1', 'MIonndur_ch1',
       'I_call_qty_all', 'callqty_w3', 'm1actdtr', 'rev_all',
       'days_main_device', 'm2actdcon', 'I_onnet_all', 'MA_Ionndur1',
       'm2diffchrgtr', 'Mmou_ch2', 'm1ratiochrgtr', 'm3diffchrgtr',
       'MIavgcalldur_ch2', 'MA_Ionnqty1', 'charge_all', 'Mavgcall_ch1',
       'MIavgcalldur_ch1', 'MIcalldur_ch2', 'MIonndur_ch2', 'MA_Ionnqty2',
       'Mmou_ch1', 'I_offnet_all', 'Ionndur_w4', 'MA_Ionndur3',
       'Monnet_ch1', 'MA_Ionndur2', 'MA_callqty1', 'm1inactdays',
       'm3ratiochrgtr', 'm2ratiochrgtr', 'offnet_all', 'm2actdtr',
       'count_all', 'MA_CON2', 'MIoffdur_ch2', 'm3actdcon', 'm3inactdays',
       'Ionndur_w3', 'rech_w1', 'MIoffdur_ch1', 'Msms_ch2', 'm2inactdays',
       'MA_rev2', 'MA_callqty3', 'count_w2', 'callqty_w4', 'MA_callqty2',
       'GPRS_w1', 'MA_MOU2', 'm3vivadays', 'count_w1', 'rev_w11',
       'GPRS_w9', 'GPRS_w8', 'rev_w4', 'MA_MOU1', 'GPRS_w4', 'rev_w3',
       'MA_CON1']
def join_split_train_test(data,tag,selected_features=selected_features):
    """This function gets the data jions it and then split it in x and y for training and testing 
    data: are lists of dataframes that will be joined
    tag= target column/split in the data 
    
    return:
    X= The feature columns
    y= the targer column
    
    """
    data=pd.concat(data)
    target=tag
    #numeric_var=list(data.loc[:, data.columns != tag].columns)
    numeric_var=selected_features
    train,test=train_test_split(data, test_size=0.25, random_state=42)
    x_train=train[numeric_var]
    y_train=train[target]
    x_test=test[numeric_var]
    y_test=test[target]
    return(x_train,y_train,x_test,y_test,data)

def predictions(X,model):
    """this function loads the model in a pipline and gives predictions 
    X= the feature columns
    model= the loaded model
    
    return=predicted_target,predicted_target_prob

    """
    predicted_target=model.predict(X)
    predicted_target_prob=model.predict_proba(X)
    #predicted_target_prob=predicted_target_prob[:,1]
    
    return predicted_target,predicted_target_prob




def model_performance(y,predicted_target,predicted_target_prob):
    """this function prints and returns the model performance
    y= the target column
    predicted_target=the predicted y
    predicted_target_prob= the predicted probability of y
    
    return acc,roc_auc,f1,ap,sen,spc
    """
    acc=accuracy_score(y,predicted_target)
    roc_auc=roc_auc_score(y, predicted_target_prob)
    cm=confusion_matrix(y,predicted_target)
    sen=cm[1,1]/(cm[1,1]+cm[1,0])
    spc=cm[0,0]/(cm[0,0]+cm[0,1])
    ppvr=cm[1,1]/(cm[1,1]+cm[0,1])
    #print('accu=%.3f roc_auc=%.3f Sensitivity=%.3f Specificity=%.3f' % (acc,roc_auc,sen,spc))
    return([acc,roc_auc,sen,spc,ppvr])

def plot_var(y,predicted_target_prob):
    """
    This funtion returns the variables for the ROC, precision recall curves
    y= the target column
    predicted_target_prob= the predicted probability of y
    """
    fpr, tpr, thresholds = roc_curve(y, predicted_target_prob)
    precision, recall, thresholds = precision_recall_curve(y, predicted_target_prob)
    return(fpr, tpr,precision, recall)




# Dict for all the classified models and there hyper parameters which will be used during gridsearchcv  
classification_models={'RF':RandomForestClassifier(random_state=42,n_estimators=100),
                       'RF_param_grid': { 'max_features': ['auto', 'sqrt', 'log2'],
                                          'max_depth' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                                          'criterion' :['gini', 'entropy'],
                                          'class_weight':[None,'balanced']},
                       'RF_name':'Random Forest',
                       'DT':DecisionTreeClassifier(random_state=42),
                       'DT_param_grid':{'criterion':['gini','entropy'],
                                        'splitter':['best','random'],
                                        'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                                        'max_features':['sqrt','log2'],
                                        'class_weight':[None,'balanced']},
                       'DT_name':'Decision Tree',
                       'GB':GradientBoostingClassifier(random_state=42,n_estimators=100),
                       'GB_param_grid':{'loss':['deviance','exponential'],
                                        'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                                        'max_features':['sqrt','log2','auto']},
                       'GB_name':'Gradient Boosting', 
                       'LR':LogisticRegression(random_state=42),
                       'LR_param_grid':{'penalty' : ['l2','none'],
                                        'C':np.logspace(-4, 4, 20),
                                        'solver': ['liblinear']},
                       'LR_name':'Logistic Regression',                 
                       'SVC':SVC(random_state=42),
                       'SVC_param_grid':{'C':np.logspace(-4, 4, 20),
                                         'kernal':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                                         'class_weight':[None,'balanced']
                                        },
                       'SVC_name':'SVC',
                       'KNN':KNeighborsClassifier(),
                       'KNN_param_grid':{'n_neighbors':[1,2,3,4,5,6,7,8,9,10],
                                         'weights':['uniform','distance'],
                                         'algorithm':['auto','ball_tree','kd_tree','brute'],
                                         'leaf_size':[10,20,30,40,50],
                                         'p':[1,2]},
                       'KNN_name':'K Neighbors',
       }

def value_range(data,prob_col):
    """
    this function is used to create the ranges column 
    data: the dataframe with prob from the models and the MSISDN
    prob_col: name of the column

    return: conditions and choices

    """
    conditions = [
    (data[prob_col] > 0) & (data[prob_col] <= 0.1),
    (data[prob_col] > 0.1) & (data[prob_col] <= 0.2),
    (data[prob_col] > 0.2) & (data[prob_col] <= 0.3),
    (data[prob_col] > 0.3) & (data[prob_col] <= 0.4),
    (data[prob_col] > 0.4) & (data[prob_col] <= 0.5),
    (data[prob_col] > 0.5) & (data[prob_col] <= 0.6),
    (data[prob_col] > 0.6) & (data[prob_col] <= 0.7),
    (data[prob_col] > 0.7) & (data[prob_col] <= 0.8),
    (data[prob_col] > 0.8) & (data[prob_col] <= 0.9),
    (data[prob_col] > 0.9) & (data[prob_col] <= 1)]
    choices = ['(0-0.1]', '(0.1-0.2]', '(0.2-0.3]','(0.3-0.4]','(0.4-0.5]','(0.5-0.6]','(0.6-0.7]','(0.7-0.8]','(0.8-0.9]','(0.9-1]']
    return(conditions,choices)

def creating_model(x,y,cv,model,param,save_model=None,saved_model_name=None):
    """
    this function gives us the model after gridsearch and prints the best parameters of thet moel
    x: features 
    y: target
    cv: how many batches the data are divided, generally either 3 or 5
    save_model: yes to save None for don't 
    saved_model_name: name of the model saved
    model:classification model or other models
    param:dict of parameters for the model

    returns: cv_model whis is the fitted model
 
    """
    name=saved_model_name
    cv_model=GridSearchCV(model,param_grid=param,cv=cv)
    cv_model.fit(x,y)
    print(cv_model.best_params_)
    print(round(cv_model.best_score_,4))
    if save_model=='yes':
        pickle.dump(cv_model,open(name,'wb'))
    else:
        print('model is not saved')
    return(cv_model)


def plot_function(y,predicted_target,predicted_target_prob,fpr,tpr,recall,precision,title,choose_plot=None,normalize=True):
    """
    This Function for plotint the resluts
    y:target column
    predicted_target: predicted values of target column
    predicted_target_prob: probability values of target column
    fpr,tpr,recall,precision: for the roc and precision_recall_curve
    title: name of the model
    choose_plot: select which plot you want("cm","roc","hist")
    normalize: either True or False if True it normalize the cm if not it gives the count 

    return: the plot/s 

    """

    

    if choose_plot=='cm':
        skplt.metrics.plot_confusion_matrix(y,predicted_target,title=title,normalize=normalize)
    elif choose_plot=='roc':
        fig, (axs1,axs2) = plt.subplots(1, 2,gridspec_kw={'hspace':0.5},figsize=(15,15));
        axs1.plot(fpr, tpr, marker='.',color='green');
        axs1.set_title('ROC Curve '+title);
        axs2.plot(recall, precision, marker='.',color='orange');
        axs2.set_title('Precision-Recall Curves '+title);
    elif choose_plot =='hist':
        plt.hist(predicted_target_prob, 10, facecolor='blue', alpha=0.5)
        plt.set_title('Probability Histogram of '+title);
    else:
        print('choose a plot')

def all_model_performance_values(mp_1,mp_2,mp_3,model_names):
    """this function give the performance of all the the models to datafarme
    mp 1-3: list of values of the performance of 3 models 
    model_names: list of names of the models 

    """
    model_pref={model_names[0]:mp_1,model_names[1]:mp_2,model_names[2]:mp_3}
    model_pref=pd.DataFrame(data=model_pref).rename(index={0:'Accuracy',1:'ROC_AUC',2:'Sensitivity',3:'Specificity',4:'Positive Predicted Value Ratio'})
    return(model_pref)

def load_saved_model(location,model_name):
    """
    this function loades the saved models

    location: the directory of the saved model
    model_name: name of the model saved 

    return: the saved model

    """
    model=pickle.load(open(location+model_name+'.sav','rb'))
    return(model)