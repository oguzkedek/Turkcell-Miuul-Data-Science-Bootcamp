import warnings

warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report, \
    roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

from scipy.stats import jarque_bera
from scipy.stats import normaltest

from helpers.data_prep import *
from helpers.eda import *

# Keşifçi veri analizi ve veri check etme gibi işlemler için oluşturduğum bazı fonksiyonları fonksiyon kalabalığı
# olmaması adına dışarıdan okutacağım.

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# %%

#######################################
# GÖREV 1 - KEŞİFÇİ VERİ ANALİZİ
#######################################

df = pd.read_csv('datasets/Telco-Customer-Churn.csv')

data_understanding(df)

"""
- Veri seti 7043 gözlem ve 21 değişkenden oluşmaktadır.
- Müşterilerin %16.2si yaşlıdır. 
- Müşterilerin şirkette kaldığı ay ortalaması 32.271'dir.
- Müşterilerden aylık olarak tahsil edilen tutar ortalaması 64.762'dir.
- Verisetinde boş değer bulunmamaktadır.
- 21 değişkenden 18'i object, 3'ü nümerik veri türündedir. Veri tipleri ile ilgili bazı hatalar mevcuttur.
"""

# Adım 1

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Customer ID cat_but_car sınıfındandır ve drop edilebilir. Total Charges değişkeni nümerik veri tipine dönüştürülecektir.
# SeniorCitizen değişkeni nümerik görünmesine rağmen object veri tipindedir.

# Adım 2
df.drop(['customerID'], axis=1, inplace=True)

for col in ['TotalCharges']:
    print("Incorrect values for {} column : ".format(col))
    incorrect_values = []
    for value in df[col]:
        try:
            float(value)
        except:
            incorrect_values.append(value)
    print(set(incorrect_values))

df[df['TotalCharges'] == ' '].shape[0]  # 11 gözlemde hata bulunmakta.
df['TotalCharges'] = df['TotalCharges'].apply(str.strip).replace('', np.nan)
# df = df[df['TotalCharges'] != " "]

df["TotalCharges"] = df["TotalCharges"].astype(float)
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# %%
# Adım 3-4

plt.figure(figsize=(12, 6))
df.groupby('Churn')['Churn'].count().plot(kind='pie', autopct='%1.1f%%')
plt.title('Churn rate', fontsize=15)
plt.show()

cats = [col for col in cat_cols if 'Churn' not in col]
for col in cats:
    cat_summary_2(df, 'Churn', col)

plt.figure(figsize=(25, 25))
for i, col in enumerate(cats):
    plt.subplot(4, 4, i + 1)
    ax = sns.countplot(x=col, data=df)
    plt.title(f"Count Analysis of {col} Features", fontsize=12)
plt.show()

plt.figure(figsize=(25, 25))
for i, col in enumerate(cats):
    plt.subplot(4, 4, i + 1)
    ax = sns.barplot(x=col, y='Churn', data=df)
    plt.title(f"Churn Analysis of {col} Features", fontsize=12)
plt.show()

"""
- Kategorik değişkenlerde Rare sayılabilecek veri bulunmamaktadır.
- PaymentMethod, Contract, StreamingMovies, StreamingTV, TechSupport , DeviceProtection, OnlineBackup, OnlineSecurity ,
InternetService, Dependents , Partner değişkenleri kırılımında Churn oranı farklılıkları göze çarpmaktadır.
- Sözleşme süresi uzadıkça churn oranı düşmektedir. 
"""
num_cols.append('TotalCharges')

# %%

# Nümerik Değişken Analizi

for col in num_cols:
    num_summary(df, 'Churn', col)

plt.figure(figsize=(20, 10))
for i, col in enumerate(num_cols):
    plt.subplot(1, 3, i + 1)
    ax = sns.histplot(data=df, x=col, kde=True)
    plt.title(f"Distribution Analysis of {col} Features", fontsize=12)
plt.show()

df[num_cols].describe([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

distribution_tests = pd.DataFrame(columns=['feature', 'jarque_bera_stats', 'jarque_bera_p_value',
                                           'normal_stats', 'normal_p_value'])
for col in num_cols:
    jb_stats = jarque_bera(np.log1p(df[col]))
    norm_stats = normaltest(np.log1p(df[col]))
    distribution_tests = distribution_tests.append({"feature": col,
                                                    "jarque_bera_stats": jb_stats[0],
                                                    "jarque_bera_p_value": jb_stats[1],
                                                    "normal_stats": norm_stats[0],
                                                    "normal_p_value": norm_stats[1]
                                                    }, ignore_index=True)
distribution_tests

#%%

# Adım 5

plt.figure(figsize=(20, 10))
for i, col in enumerate(num_cols):
    plt.subplot(1, 3, i + 1)
    ax = sns.boxplot(data=df, x=col)
    plt.title(f"Outlier Analysis of {col} Features", fontsize=12)
plt.show()

for col in num_cols:
    print(col, check_outlier(df, col))

# Aykırı gözlem bulunmamaktadır.

#%%

# Adım 6

df.isnull().sum()

# TotalCharges değişkeninde 11 adet eksik veri bulunmaktadır.

#######################################
# GÖREV 2 - FEATURE ENGINEERING
#######################################

# Adım 1

df[df['TotalCharges'].isnull()]
df.loc[df['TotalCharges'].isnull(), 'TotalCharges'] = 0

# Adım 2

df['total_charges_level'] = pd.qcut(df['TotalCharges'], 5, labels=['low', 'low_med', 'medium', 'med_high', 'high'])
df['monthly_level'] = pd.qcut(df['MonthlyCharges'], 5, labels=['low', 'low_med', 'medium', 'med_high', 'high'])
df['tenure_level'] = pd.qcut(df['tenure'], 5, labels=['low', 'low_med', 'medium', 'med_high', 'high'])

df.loc[(df['monthly_level'].isin(['high', 'med_high'])) & (df['tenure_level'] == 'high'), 'cust_seg'] = 'high_income'
df.loc[(df['monthly_level'].isin(['low_med', 'medium'])) & (df['tenure_level'] == 'high'), 'cust_seg'] = 'med_inc_loyal'
df.loc[(df['monthly_level'] == 'low') & (df['tenure_level'] == 'high'), 'cust_seg'] = 'low_inc_loyal'
df.loc[(df['monthly_level'].isin(['high', 'med_high'])) &
       (df['tenure_level'].isin(['med_high', 'medium'])), 'cust_seg'] = 'pay_attention'
df.loc[(df['monthly_level'].isin(['low_med', 'medium'])) &
       (df['tenure_level'].isin(['med_high', 'medium'])), 'cust_seg'] = 'standard'
df.loc[(df['monthly_level'] == 'low') & (df['tenure_level'].isin(['med_high', 'medium'])), 'cust_seg'] = 'not_pot_churn'
df.loc[(df['monthly_level'].isin(['high', 'med_high'])) &
       (df['tenure_level'].isin(['low_med', 'low'])), 'cust_seg'] = 'more_pot_churn'
df.loc[(df['monthly_level'].isin(['low_med', 'medium'])) &
       (df['tenure_level'].isin(['low_med', 'low'])), 'cust_seg'] = 'ave_pot_churn'
df.loc[(df['monthly_level'] == 'low') & (df['tenure_level'].isin(['low_med', 'low'])), 'cust_seg'] = 'indecisive'

df.groupby('cust_seg').agg({'Churn': ['count', 'mean']})

#               Churn
#               count  mean
# cust_seg
# ave_pot_churn   1417 0.435
# high_income      824 0.100
# indecisive       688 0.174
# low_inc_loyal    212 0.005
# med_inc_loyal    371 0.027
# more_pot_churn   773 0.665
# not_pot_churn    520 0.017
# pay_attention   1218 0.307
# standard        1020 0.139

df['Service_Count'] = (df[['PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                           'StreamingTV', 'StreamingMovies']] == "Yes"). \
    sum(axis=1)

df.groupby('SeniorCitizen').Churn.mean()  # 0 : 0.236, 1: 0.417

df.groupby(['SeniorCitizen', 'PaperlessBilling']).Churn.mean()
# 0              No                 0.150
#                Yes                0.304
# 1              No                 0.293
#                Yes                0.454

df.groupby('PaperlessBilling').Churn.mean()  # No: 0.163, Yes: 0.336

df.loc[(df['PaperlessBilling'] == 'No') & (df['SeniorCitizen'] == 0), 'paperless_senior'] = 'not_paperless_junior'
df.loc[(df['PaperlessBilling'] == 'Yes') & (df['SeniorCitizen'] == 0), 'paperless_senior'] = 'paperless_junior'
df.loc[(df['PaperlessBilling'] == 'No') & (df['SeniorCitizen'] == 1), 'paperless_senior'] = 'not_paperless_senior'
df.loc[(df['PaperlessBilling'] == 'Yes') & (df['SeniorCitizen'] == 1), 'paperless_senior'] = 'paperless_senior'

df.groupby(['SeniorCitizen', 'Partner']).Churn.mean()

# SeniorCitizen  Partner
# 0              No        0.300
#                Yes       0.166
# 1              No        0.489
#                Yes       0.346

df.loc[(df['Partner'] == 'No') & (df['SeniorCitizen'] == 0), 'partner_senior'] = 'alone_junior'
df.loc[(df['Partner'] == 'Yes') & (df['SeniorCitizen'] == 0), 'partner_senior'] = 'not_alone_junior'
df.loc[(df['Partner'] == 'No') & (df['SeniorCitizen'] == 1), 'partner_senior'] = 'alone_senior'
df.loc[(df['Partner'] == 'Yes') & (df['SeniorCitizen'] == 1), 'partner_senior'] = 'not_alone_senior'


def tenure_year(tenure):
    if tenure <= 12:
        return '0-1'
    elif tenure <= 24:
        return '1-2'
    elif tenure <= 48:
        return '2-4'
    else:
        return '4-..'


df['tenure_year'] = df['tenure'].apply(tenure_year)

# df.groupby('tenure_year').Churn.mean()
# tenure_year
# 0-1    0.474
# 1-2    0.287
# 2-4    0.204
# 4-..   0.095

# Adım 3

#%%

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

df.shape  # 7043 , 67
df.head()

df.dtypes

#%%

# Adım 4

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

#######################################
# GÖREV 3 - MODELLEME
#######################################

# Adım 1

y = df["Churn"]
X = df.drop(["Churn"], axis=1)

log_model = LogisticRegression()
cart_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()
rf_model = RandomForestClassifier()
gbm_model = GradientBoostingClassifier()
xgboost_model = XGBClassifier()
lgbm_model = LGBMClassifier()

models = [log_model, cart_model, knn_model, rf_model, gbm_model, xgboost_model, lgbm_model]

model_scores = pd.DataFrame(columns=['accuracy_mean', 'f1_mean', 'roc_auc_mean'])

for model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
    model_scores = model_scores.append({'accuracy_mean': cv_results['test_accuracy'].mean(),
                                        'f1_mean': cv_results['test_f1'].mean(),
                                        'roc_auc_mean': cv_results['test_roc_auc'].mean()}, ignore_index=True)

model_scores['models'] = ['logistic', 'cart', 'knn', 'rf', 'gbm', 'xgboost', 'lgbm']
model_scores.sort_values(by='accuracy_mean', ascending=False)

# logistic , gbm , lgbm ve rf accuracy skorlarına göre en başarılı 4 algoritmadır.

#%%

# Adım 2


# Logistic Regression

parameters = {"C": [10 ** x for x in range(-10, 10)],
              "penalty": ['l1', 'l2']
              }
log_best_grid = GridSearchCV(estimator=log_model,
                             param_grid=parameters,
                             cv=5
                             ).fit(X, y)

print("Best Parameters : ", log_best_grid.best_params_)
# {'C': 0.1, 'penalty': 'l2'}
log_final = log_model.set_params(**log_best_grid.best_params_).fit(X, y)

# GBM
gbm_model.get_params()

gbm_params = {'learning_rate': [0.01, 0.05, 0.1, 0.25],
              'max_depth': [3, 5, 7, 10],
              'n_estimators': [100, 500, 1000],
              'subsample': [1, 0.5, 0.7],
              'min_samples_split': [2, 4, 8, 10]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=3, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_
# {'learning_rate': 0.05, 'max_depth': 3, 'min_samples_split': 8, 'n_estimators': 100, 'subsample': 0.5}

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_).fit(X, y)

# LGBM

lgbm_model.get_params()

lgbm_params = {'n_estimators': [100, 300, 500, 1000],
               'colsample_bytree': [0.5, 0.7, 1],
               'subsample': [0.6, 0.8, 1.0],
               'max_depth': [3, 4, 5, 6],
               'learning_rate': [0.1, 0.01, 0.05],
               'min_child_samples': [5, 10, 20]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_best_grid.best_params_
# {'colsample_bytree': 0.5, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_samples': 5,
# 'n_estimators': 300, 'subsample': 0.6}

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_).fit(X, y)

# RF

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=3, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_
# {'max_depth': None, 'max_features': 'auto', 'min_samples_split': 20, 'n_estimators': 200}

rf_final = rf_model.set_params(**rf_best_grid.best_params_).fit(X, y)

final_models = [log_final, gbm_final, lgbm_final, rf_final]
final_scores = pd.DataFrame(columns=['accuracy_mean', 'f1_mean', 'roc_auc_mean'])

for model in final_models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
    final_scores = final_scores.append({'accuracy_mean': cv_results['test_accuracy'].mean(),
                                        'f1_mean': cv_results['test_f1'].mean(),
                                        'roc_auc_mean': cv_results['test_roc_auc'].mean()}, ignore_index=True)

final_scores['models'] = ['logistic', 'gbm', 'lgbm', 'rf']
final_scores.sort_values(by='accuracy_mean', ascending=False)


#  accuracy_mean f1_mean roc_auc_mean    models
# 0         0.808   0.594        0.848  logistic
# 2         0.806   0.588        0.848      lgbm
# 1         0.804   0.577        0.848       gbm
# 3         0.802   0.569        0.842        rf

# accuracy_mean f1_mean roc_auc_mean    models
# 0         0.807   0.596        0.848  logistic
# 4         0.804   0.586        0.847       gbm
# 6         0.797   0.578        0.836      lgbm
# 3         0.794   0.559        0.823        rf
# 5         0.785   0.556        0.825   xgboost
# 2         0.775   0.558        0.777       knn
# 1         0.725   0.498        0.659      cart

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(lgbm_final, X, save=True)
plot_importance(gbm_final, X)

final_cols = X.columns[lgbm_final.feature_importances_ >= 20]
X_final = X[final_cols]

# LGBM

lgbm_best_grid_final = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_final, y)

lgbm_best_grid_final.best_params_

# {'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 3,
# 'min_child_samples': 10, 'n_estimators': 100, 'subsample': 0.6}

lgbm_final_2 = lgbm_model.set_params(**lgbm_best_grid_final.best_params_).fit(X_final, y)

cv_results_final = cross_validate(lgbm_final_2, X_final, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results_final['test_accuracy'].mean()
# 0.807
cv_results_final['test_f1'].mean()
# 0.5925
cv_results_final['test_roc_auc'].mean()
# 0.847
