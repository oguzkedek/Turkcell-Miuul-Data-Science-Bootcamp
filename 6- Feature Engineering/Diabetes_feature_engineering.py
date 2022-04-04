import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import jarque_bera
from scipy.stats import normaltest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import GradientBoostingClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv('datasets/diabetes.csv')

###########################################
# Görev 1 - Keşifçi Veri Analizi
###########################################

df.head()
df.info()
df.shape
df.describe([.1, .25, .75, .9, .95]).T
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

for col in df.columns:
    print(f"Count number of distinct elements in {col} : {df[col].nunique()}")

"""
- Veri seti 768 gözlem ve 9 değişkenden oluşmaktadır.
- Eksik veri bulunmamaktadır. 
- Betimsel istatistiklere baktığımızda hatalı değerlerin varlığı göze çarpıyor. Glikoz , insulin
gibi değişkenlerin minimum değerlerinin 0 olması eksik veya hatalı veri anlamına gelmektedir. 
"""


# %%

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Outcome değişkeni nümerik gibi görünse de kategorik özellikte bir değişkendir.


# %%


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("=" * 50)
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col, plot=True)


# Outcome değişkeninin %65.1'i 0 , %34.9'u 1 değerinden oluşmaktadır.

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col)
    print('=' * 50)

sns.pairplot(df)
plt.show()

distribution_tests = pd.DataFrame(columns=['feature', 'jarque_bera_stats', 'jarque_bera_p_value',
                                           'normal_stats', 'normal_p_value'])

for col in num_cols:
    jb_stats = jarque_bera(np.log(df[col] + 1))
    norm_stats = normaltest(np.log(df[col] + 1))
    distribution_tests = distribution_tests.append({'feature': col,
                                                    'jarque_bera_stats': jb_stats[0],
                                                    'jarque_bera_p_value': jb_stats[1],
                                                    'normal_stats': norm_stats[0],
                                                    'normal_p_value': norm_stats[1]
                                                    }, ignore_index=True)

distribution_tests


# Nümerik verilerin dağılımı normal dağılıma uymamaktadır.

# %%


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, 'Outcome', col)


# %%

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(f'Outlier check {col} : {check_outlier(df, col)}')

# Aykırı değer bulunmamaktadır.

# %%

corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.3f', cmap="RdBu")
plt.show()

# Glucose değişkeni hedef değişken ile en yüksek korelasyona sahip değişkendir.

###########################################
# Görev 2 - Feature Engineering
###########################################

num_cols_np = [col for col in num_cols if col not in 'Pregnancies']
for col in num_cols_np:
    df.loc[df[col] == 0, col] = np.nan

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

"""
missing_values_table(df)
               n_miss  ratio
Insulin           374 48.700
SkinThickness     227 29.560
BloodPressure      35  4.560
BMI                11  1.430
Glucose             5  0.650
"""

df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

df.loc[(df['Glucose'] < 53), 'glucose_level'] = 'very_low'
df.loc[(df['Glucose'] >= 53) & (df['Glucose'] < 70), 'glucose_level'] = 'low'
df.loc[(df['Glucose'] >= 70) & (df['Glucose'] < 125), 'glucose_level'] = 'normal'
df.loc[(df['Glucose'] >= 125), 'glucose_level'] = 'high'

df.loc[(df['Pregnancies'] > 0), 'ever_pregnant'] = 'yes'
df.loc[(df['Pregnancies'] == 0), 'ever_pregnant'] = 'no'

df.loc[(df['BMI'] < 18.5), 'bmi_level'] = 'underweight'
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 25), 'bmi_level'] = 'normal_weight'
df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30), 'bmi_level'] = 'overweight'
df.loc[(df['BMI'] >= 30) & (df['BMI'] < 35), 'bmi_level'] = 'obesity_class_1'
df.loc[(df['BMI'] >= 35), 'bmi_level'] = 'obesity_class_2'

df.loc[(df['Age'] < 18), 'age_level'] = 'teen'
df.loc[(df['Age'] >= 18) & (df['Age'] < 25), 'age_level'] = 'young'
df.loc[(df['Age'] >= 25) & (df['Age'] < 40), 'age_level'] = 'young_adults'
df.loc[(df['Age'] >= 40) & (df['Age'] < 60), 'age_level'] = 'middle_aged_adults'
df.loc[(df['Age'] >= 60), 'age_level'] = 'old_adults'

df.head()

#%%

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()

#%%

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

#%%

y = df['Outcome']
X = df.drop(['Outcome'], axis=1)

gbm_model = GradientBoostingClassifier(random_state=17)

cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7747304982599099
cv_results['test_f1'].mean()
# 0.664032028737911
cv_results['test_roc_auc'].mean()
# 0.8353347309573724


