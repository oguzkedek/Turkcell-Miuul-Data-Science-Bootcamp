import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

from helpers.data_prep import *
from helpers.eda import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

train = pd.read_csv('Hafta_10/demand_forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('Hafta_10/demand_forecasting/test.csv', parse_dates=['date'])

# %%

df = pd.concat([train, test], sort=False)

df["date"].min(), df["date"].max()

# (Timestamp('2013-01-01 00:00:00'), Timestamp('2018-03-31 00:00:00'))

data_understanding(df)

# Veri seti 958023 gözlem ve 5 değişkenden oluşmaktadır.
# Veri setinde % 95.30 oranında id değişkeninde null değer vardır. Bu null değerler train test setini temsil etmektedir.
# % 4.70 oranında sales değişkeninde var olan null değerler ise test setini temsil etmektedir.

df.groupby(["store"])["item"].nunique()
df.groupby(["store", "item"]).agg({"sales": ["sum"]})

# Satışların store ve itemlara göre değişiklik gösterdiği gözlenmektedir.



# %%

# FEATURE ENGINEERING

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df['is_wknd'] = df.date.dt.weekday // 4
    df['season'] = np.where(df.month.isin([12, 1, 2]), 0, 1)
    df['season'] = np.where(df.month.isin([6, 7, 8]), 2, df['season'])
    df['season'] = np.where(df.month.isin([9, 10, 11]), 3, df['season'])
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)

df.info()
new_cols = df.iloc[:, 5:].columns

for col in new_cols:
    print(df.groupby(col).agg({'sales': ['count', 'sum', 'mean', 'median']}))
    print('='*50)

plt.figure(figsize=(30, 25))
for i, col in enumerate(new_cols):
    plt.subplot(5, 3, i+1)
    ax = sns.barplot(y='sales', x=col, data=df)
    plt.title(f"Sales Analysis of {col} Features", fontsize=12)
plt.show()

"""
- Aylara göre satış incelemesi yapıldığında Ocak,Şubat ve Aralık ayındaki satışlar en düşük düzeydedir. Kış mevsiminin 
aksine Yaz aylarındaki satışlar ise en yüksek seviyelerdedir. 
- Dolayısıyla 1.çeyrekteki satışlar da en düşük seviyedeyken , 2  ve 3. çeyrekler birbirine oldukça yakın olmakla birlikte 
en yüksek satışın yapıldığı çeyreklerdir.
- Çeyreklerin son günlerindeki satış ortalaması 50.75 iken diğer dönemlerde satış ortalaması 52.25'tir. 
- Haftanın günleri olarak bakıldığında Pazar günü en çok satışın yapıldığı gündür. Pazartesi ise en az satışın yapıdığı gündür.
- Satışlarda yıllara göre yükselen trend bulunmaktadır. 
"""

# %%

# Random Noise
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

# Lag/Shifted Features

df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

pd.DataFrame({"sales": df["sales"].values[0:10],
              "lag1": df["sales"].shift(1).values[0:10],
              "lag2": df["sales"].shift(2).values[0:10],
              "lag3": df["sales"].shift(3).values[0:10],
              "lag4": df["sales"].shift(4).values[0:10]})
"""
   sales  lag1  lag2  lag3  lag4
0   13.0   NaN   NaN   NaN   NaN
1   13.0  13.0   NaN   NaN   NaN
2   11.0  13.0  13.0   NaN   NaN
3   11.0  11.0  13.0  13.0   NaN
4   14.0  11.0  11.0  13.0  13.0
5   14.0  14.0  11.0  11.0  13.0
6   13.0  14.0  14.0  11.0  11.0
7   13.0  13.0  14.0  14.0  11.0
8   10.0  13.0  13.0  14.0  14.0
9   10.0  10.0  13.0  13.0  14.0
"""

# Geçmiş gerçek değerlerden yeni featurelar üretiyoruz.

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 114, 120, 128, 182, 364, 547, 729])

data_understanding(df)

# %%

# Rolling Mean Features

# Hareketli ortalamayı ekliyoruz. Shift 1 yaparak o günden öncesinin hareketli ortalamasını alıyoruz.

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})

"""
   sales  roll2      roll3  roll5
0   13.0    NaN        NaN    NaN
1   13.0    NaN        NaN    NaN
2   11.0   13.0        NaN    NaN
3   11.0   12.0  12.333333    NaN
4   14.0   11.0  11.666667    NaN
5   14.0   12.5  12.000000   12.4
6   13.0   14.0  13.000000   12.6
7   13.0   13.5  13.666667   12.6
8   10.0   13.0  13.333333   13.0
9   10.0   11.5  12.000000   12.8

"""


def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [364, 546])

# %%
# Exponentially Weighted Mean Features

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm02": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})

"""
   sales  roll2     ewm099     ewm095      ewm07      ewm02
0   13.0    NaN        NaN        NaN        NaN        NaN
1   13.0    NaN  13.000000  13.000000  13.000000  13.000000
2   11.0   13.0  13.000000  13.000000  13.000000  13.000000
3   11.0   12.0  11.019998  11.099762  11.561151  12.261993
4   14.0   11.0  11.000200  11.004988  11.165138  11.895028
5   14.0   12.5  13.970002  13.850250  13.154375  12.409050
6   13.0   14.0  13.999700  13.992513  13.746744  12.748591
7   13.0   13.5  13.009997  13.049626  13.223909  12.796781
8   10.0   13.0  13.000100  13.002481  13.067162  12.832463
9   10.0   11.5  10.030001  10.150124  10.920106  12.370080
"""

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)
data_understanding(df)

#%%
# One-Hot Encoding

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'],
                    drop_first=True)

#%%

df['sales'] = np.log1p(df['sales'].values)


#%%
# MODEL

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

# 2017'nin başına kadar (2016'nın sonuna kadar) train seti.
train = df.loc[(df['date'] < '2017-01-01'), :]

# 2017'nin ilk 3'ayı validasyon seti.
val = df.loc[(df['date'] >= '2017-01-01') & (df['date'] < '2017-04-01'), :]

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape
#((730523,), (730523, 155), (45000,), (45000, 155))

#%%

# LightGBM ile Zaman Serisi Modeli

# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 12000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=200)

#Early stopping, best iteration is:
#[9931]	training's SMAPE: 12.7576	valid_1's SMAPE: 13.4848


y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))
#13.4848085

#%%

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=30, plot=True)

feat_imp = plot_lgb_importances(model, num=200)
feat_imp[:20]
"""
                         feature  split       gain
18           sales_roll_mean_546   7354  60.074762
14                 sales_lag_364   5884  11.442233
17           sales_roll_mean_364   5401   5.375587
79    sales_ewm_alpha_04_lag_365   1691   4.608518
4                         season   1498   4.086553
7                   sales_lag_91   1995   2.157718
3                        is_wknd   1304   1.684635
46     sales_ewm_alpha_07_lag_91    367   1.207188
1                    day_of_year   3982   1.052515

"""

useless_features = feat_imp[feat_imp["gain"] < 0.001]["feature"].values
imp_feats = [col for col in cols if col not in useless_features]
len(imp_feats)
#143

#%%

# Final Model

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

#%%

submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submission_demand.csv", index=False)
