import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import jarque_bera
from scipy.stats import normaltest

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df = pd.read_csv("datasets/hitters.csv")

#%%

def data_understanding(df, percentiles=[.01, .05, .10,.25,.5,.75,.9, .95, .99]):
    print('\033[1m', 'İlk 10 Gözlem', '\033[0m', df.head(10), sep='\n')
    print('-'*150)
    print('\033[1m', 'Değişken İsimleri', '\033[0m', df.columns, sep='\n')
    print('-' * 150)
    print('\033[1m', "Veri Shape'i" , '\033[0m', df.shape, sep='\n')
    print('-' * 150)
    print('\033[1m', 'Betimsel İstatistik', '\033[0m', df.describe(percentiles=percentiles).T, sep='\n')
    print('-' * 150)
    print('\033[1m', 'Boş Değer Check', '\033[0m', (df.isnull().sum() * 100 / len(df)). \
          sort_values(ascending=False), sep='\n')
    print('-' * 150)
    print('\033[1m', 'Değişken Tipleri İncelemesi', '\033[0m')
    print(df.info())

data_understanding(df)

"""
-  322 gözlem ve 20 değişkeni olan bir veri setine sahibiz. 
- Veri setinde 1 float64, 16 int64 ve 3 object veri tipinde değişken bulunmaktadır.
- Hedef değişkenimiz 'salary' değişkeni sayısal veri tipine sahiptir ve problem tipi regresyondur.
- Salary değişkeninde % 18.323 oranında boş değer bulunmaktadır.
- Veri tipinde ölçek farkı göze çarpmaktadır. 
- Betimsel istatistiklerine baktığımızda birçok sütunda min ve %25'lik çeyrekler ile max ve %75'lik çeyrekler 
 arasındaki fark göze çarpmaktadır. Aykırı değerleri %10 ve %90lık değerlere baskılayacağım . 
 
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

# 3 tane kategorik değişken ve 17 tane nümerik değişkenimiz bulunmaktadır.

#%%

# KATEGORİK DEĞİŞKEN ANALİZİ

def cat_summary(dataframe, target, col_name):
    print(pd.DataFrame({"Count": dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe),
                        "Salary_Mean": dataframe.groupby(col_name)[target].mean()}).sort_values(by=['Salary_Mean'],
                                                                                               ascending=False),
          end='\n\n')

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x=dataframe[col_name], data=dataframe)
    plt.title('Count Analysis of {} Features'.format(col_name), fontsize=15)
    plt.xticks(rotation=45)
    plt.subplot(1, 2, 2)
    ax = sns.boxplot(x=dataframe[col_name], y=dataframe[target])
    ax = sns.swarmplot(x=dataframe[col_name], y=dataframe[target], dodge=True, color='0.1')
    plt.title('Salary Analysis of {} Features'.format(col_name), fontsize=15)
    plt.xticks(rotation=45)
    plt.show()
    print('-' * 120)

for col in cat_cols:
    cat_summary(df, 'Salary', col)
"""
- League değişkeninde A ve N isimli iki farklı lig bulunmaktadır. Bu kırılıma göre veri setindeki gözlemlerin
%54.35'i A liginden ve % 45.65'i N ligindendir. A liginin maaş ortalaması 542k iken N liginin ortalaması 529 k'dır.

- Division kategorisine yani oyuncunun oynadığı pozisyona göre iki farklı değer yer almaktadır. E değerinin gözlemlerdeki 
bulunma oranı %48.76 iken W değerinin bulunma oranı %51.24'tür. E pozisyonunda oynayan oyuncuların maaş ortalaması
624k ve W pozisyonunda oynayan oyuncuların maaş ortalaması 450k'dır. 

- Newleague kategorisinde de iki farklı değer yer almaktadır. Veri setinde  A ligine ait oyuncuların bulunma oranı %54.658
iken N ligine ait oyuncuların bulunma oranı %45.342'dir. A liginin oyuncularının maaş ortalaması 537k ve N liginin 
oyuncularının maaş ortalaması 534k dır. 1987 yılının başında 1986 sonuna göre maaşlarda A liginde küçük bir azalma ve 
N liginde küçük bir artış olduğu söylenebilir. 

"""
plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Division', y='Salary', hue='NewLeague', data=df)
plt.subplot(1, 2, 2)
sns.barplot(x='Division', y='Salary', hue='League', data=df)
plt.show()

#%%
# NÜMERİK DEĞİŞKEN ANALİZİ

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='Salary', kde=True)
plt.show()


plt.figure(figsize=(30, 25))
for i, col in enumerate(num_cols):
    plt.subplot(4, 5, i+1)
    ax = sns.histplot(data=df, x=col, kde=True)
    plt.title('Distribution Analysis of {} Features'.format(col), fontsize=12)
plt.show()


plt.figure(figsize=(30, 25))
for i, col in enumerate(num_cols):
    plt.subplot(4, 5, i+1)
    ax = sns.boxplot(data=df, x=col)
    plt.title('Outlier Analysis of {} Features'.format(col), fontsize=12)
plt.show()

distribution_tests = pd.DataFrame(columns=['feature', 'jarque_bera_stats', 'jarque_bera_p_value',
                                         'normal_stats', 'normal_p_value'])

for col in num_cols:
    jb_stats = jarque_bera(np.log1p(df[col]))
    norm_stats = normaltest(np.log1p(df[col]))
    distribution_tests = distribution_tests.append({"feature": col,
                                                "jarque_bera_stats" : jb_stats[0] ,
                                                "jarque_bera_p_value" : jb_stats[1] ,
                                                "normal_stats": norm_stats[0] ,
                                                "normal_p_value" : norm_stats[1]
                                               }, ignore_index=True)
distribution_tests


# Nümerik değişkenler normal dağılmamaktadır ve aykırı değer bulunan sütunlar bulunmaktadır.

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, annot=True, fmt='.2f', cmap="RdBu")
plt.show()

corr_matrix = df.corr().abs()
high_corr_var = np.where(corr_matrix > 0.90)
high_corr_var = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*high_corr_var) if x != y and x < y]

""""

- Hedef değişkenimiz ile en yüksek korelasyona sahip sütunlar CRBI, CRuns, Chits değişkenleridir.
-Birbiri ile %90 üzeri korelasyona sahip değişken çiftleri aşağıdaki gibidir. 

[('AtBat', 'Hits'),
 ('AtBat', 'Runs'),
 ('Hits', 'Runs'),
 ('Years', 'CAtBat'),
 ('Years', 'CHits'),
 ('CAtBat', 'CHits'),
 ('CAtBat', 'CRuns'),
 ('CAtBat', 'CRBI'),
 ('CAtBat', 'CWalks'),
 ('CHits', 'CRuns'),
 ('CHits', 'CRBI'),
 ('CHmRun', 'CRBI'),
 ('CRuns', 'CRBI'),
 ('CRuns', 'CWalks')]

- CaTBat 5 , CHits ve  CRuns 4 farklı değişken ile yüksek korelasyon halindedir. Bu sütunlar drop edilebilir.
 
"""

df.drop(['CAtBat', 'CHits'], axis=1, inplace=True)
df.shape

# Veri setimizden 2 sütun eksilmiştir ve 18 sütun kalmıştır.

#%%

# AYKIRI DEĞER ANALİZİ


def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.9):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))

# % 10 ve % 90'lık çeyreklere göre baktığımızda CHmRun , CWalks sütunlarında aykırı gözlemler bulunmaktadır.

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

# Aykırı değerleri belirlediğimiz alt ve üst limitlere baskıladık .

#%%

# EKSİK DEĞER ANALİZİ

# Not = Salary değişkenimiz hedef değişkenimiz olduğu için eksik değerleri doldurulmayabilir.

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

df.groupby(['Division', 'NewLeague'])['Salary'].mean()
df.groupby(['Division', 'NewLeague'])['Salary'].median()

# Oyuncu pozisyonu ve lige göre maaş ortalamalarında farklılıklar bulunmaktadır. Bu nedenle maaş değişkenindeki
# boş değerler bu kırılımlara göre doldurulacaktır. Yanlılık olmaması açısından ortalama yerine medyan ile doldurma
# yapılacaktır.

df['Salary'] = df.groupby(['Division', 'NewLeague'])['Salary'].apply(lambda x: x.fillna(x.median()))

df.isnull().any().sum()

#%%

# FEATURE ENGINEERING

df['hits_per_atbat'] = df['AtBat'] / df['Hits']  #beyzbol sopası ile vuruş sayısı / isabet sayısı
df['cruns_per_chmrun'] = df['CHmRun'] / df['CRuns'] # en değerli vuruş sayısı / kariyeri boyunca kazandırdığı sayı
df['assists_RBI'] = df['RBI'] * df['Assists'] # koşu yaptırdığı oyuncu sayısı * asist
df['walks_per_year'] = df['Walks'] / df['Years'] # yaptırılan hata / oynama süresi
df['errors_per_year'] = df['Errors'] / df['Years'] # oyuncunun hatası / oynama süresi
df['years_chmrun'] = df['Years'] * df['CHmRun'] # oynama süresi * kariyeri boyunca en değerli sayısı
df['years_cruns'] = df['Years'] * df['CRuns'] #oynama süresi * kariyeri boyunca kazandırdığı sayı
df['years_crbi'] = df['Years'] * df['CRBI']
df['years_cwalks'] = df['Years'] * df['CWalks']
df['years_putouts'] = df['Years'] * df['PutOuts']
df['experience_level'] = pd.cut(x=df['Years'],
                                bins=[0, 3, 6, 11, 25],
                                labels=['Rookie', 'average', 'experienced', 'highly_experienced']). \
                                astype("O")

df.shape

df.groupby('experience_level').Salary.mean()

"""
experience_level
Rookie               229.891
average              499.468
experienced          684.992
highly_experienced   670.623
"""

df.describe().T

""""
- Rookie seviyesindeki oyunların maaş ortalaması 229k iken average seviyesindekiler 499k, experienced 
seviyedekiler 684k ve highly experienced seviyedeki oyuncuların ortalaması 670k dır. 
"""
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


cat_cols, num_cols, cat_but_car = grab_col_names(df)

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

df.columns = [col.upper() for col in df.columns]

df.head()

#  MODELING

y = df[["SALARY"]]
X = df.drop('SALARY', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=144)
model = LinearRegression().fit(X_train, y_train)

model.intercept_
model.coef_

##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE
y_pred = model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
#0.443

# TRAIN RKARE
model.score(X_train, y_train)
# 0.693

# Test RMSE
y_pred = model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#0.653

# Test RKARE
model.score(X_test, y_test)
#0.5869

np.mean(np.sqrt(-cross_val_score(model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))
#0.553