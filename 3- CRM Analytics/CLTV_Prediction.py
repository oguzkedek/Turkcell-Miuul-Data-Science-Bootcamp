#######################
# Oğuz Kedek - Case - Endüstri Projesi 2
#######################


import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

# %%


###############################################
# Görev 1 Veriyi Hazırlama
###############################################

df_ = pd.read_csv('Hafta_3/RFM_Analizi/data_20K.csv')
df = df_.copy()

df.describe().T
df.info()


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = (quartile3 + 1.5 * interquantile_range).round()
    low_limit = (quartile1 - 1.5 * interquantile_range).round()
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Bir önceki RFM probleminde keşfettiğimiz outlier değerlerini 0.01 ve 0.99'luk değerler ile değiştireceğiz.

cols = ['order_num_total_ever_online', 'order_num_total_ever_offline',
        'customer_value_total_ever_offline', 'customer_value_total_ever_online']

for col in cols:
    replace_with_thresholds(df, col)

df['omnichannel_order_num_total'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['omnichannel_customer_value_total'] = df['customer_value_total_ever_offline'] + df[
    'customer_value_total_ever_online']

df.describe().T
df.info()

df.iloc[:, 3:7] = df.iloc[:, 3:7].apply(pd.to_datetime)

today_date = df['last_order_date'].max() + pd.DateOffset(days=2)

###############################################
# Görev 2 CLTV Veri Yapısının Oluşturulması
###############################################

cltv = pd.DataFrame()
cltv['customer_id'] = df['master_id']
cltv['recency_cltv_weekly'] = ((df['last_order_date'] - df['first_order_date']).dt.days) / 7
cltv['T_weekly'] = ((today_date - df['first_order_date']).dt.days) / 7
cltv['frequency'] = df['omnichannel_order_num_total']
cltv['monetary_cltv_avg'] = df['omnichannel_customer_value_total'] / cltv['frequency']
cltv = cltv[cltv['frequency'] > 1]

cltv.describe().T


#%%

#################################################################################
# Görev 3 BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’ninHesaplanması
#################################################################################


bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv['frequency'],
        cltv['recency_cltv_weekly'],
        cltv['T_weekly'])

cltv["expected_sales_3_month"] = bgf.predict(12,
                                             cltv['frequency'],
                                             cltv['recency_cltv_weekly'],
                                             cltv['T_weekly'])

cltv["expected_sales_6_month"] = bgf.predict(24,
                                             cltv['frequency'],
                                             cltv['recency_cltv_weekly'],
                                             cltv['T_weekly'])

cltv.sort_values(by='expected_sales_3_month', ascending=False)[['customer_id', 'expected_sales_3_month']].head(10)
cltv.sort_values(by='expected_sales_6_month', ascending=False)[['customer_id', 'expected_sales_6_month']].head(10)

# 3 aylık ve 6 aylık periyotlarda en fazla satın alım yapması beklenen 10 müşteri aynı müşterilerdir.

cltv.describe().T

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv['frequency'], cltv['monetary_cltv_avg'])

cltv['exp_average_value'] = ggf.conditional_expected_average_profit(cltv['frequency'],
                                                                    cltv['monetary_cltv_avg'])

cltv['cltv'] = ggf.customer_lifetime_value(bgf,
                                           cltv['frequency'],
                                           cltv['recency_cltv_weekly'],
                                           cltv['T_weekly'],
                                           cltv["monetary_cltv_avg"],
                                           time=6,
                                           freq='W',
                                           discount_rate=0.01)

cltv.head()

#%%

#######################################################
# Görev 4  CLTV Değerine Göre SegmentlerinOluşturulması
#######################################################

scaler = MinMaxScaler()
cltv['scaled_cltv'] = scaler.fit_transform(cltv[['cltv']])

cltv.sort_values(by='cltv', ascending=False).head(20)

"""

En fazla satın alım yapması beklenen müşteriler ile cltv değeri en yüksek müşteriler arasında farklılıkLar bulunmaktadır. 
Cltv değerlendirmesinde tek başına satın alım sayısının etkili olmadığı görülmektedir.

"""

cltv['segment'] = pd.qcut(cltv['scaled_cltv'], 4, labels=['S4', 'S3', 'S2', 'S1'])

cltv.groupby('segment').agg({'std', 'mean', 'sum'})

#%%


"""
Yorumlar
 
- Adım 2 

Segmentlerin standart sapma , ortalama , toplam gibi metrikler açısından değerlendirilmesinde 
segmentler birbirinden belirgin şekilde ayrılmaktadır. Segmentlerin kendi içinde değerlendirilmesinde 
standart sapmaların yüksek olduğu da gözlemlenmektedir. Özellikle S1 segmentinin standart sapması yüksektir 
ve bu segmenti parçalayabiliriz. KMeans ile küme içi benzerlikler maksimum ve kümeler arası benzerlikler
minimum olacak şekilde farklı bir kümeleme,segmentasyon işlemi de yapılabilir.

- Adım 3 

 S1 : S1 segmenti en yüksek cltv skoruna sahip , bizim için oldukça değerli müşterilerdir. Aynı zamanda bu müşterileri
 elde tutmak yeni müşteri kazanmaktan daha az maliyetlidir. Bu müşterilerle özel ilgilenilmesi gerekmektedir
. Sadakatin devamlılığını sağlamak için sosyal etkileşimde kalınabilir. Aynı zamanda müşteri ne kadar harcarsa o kadar 
kazanacağı şekilde bir sadakat planı geliştirebilir ve bu sayede müşterilerin devamlılığı korunabilir. Özel indirimler 
yapılabilir fakat bu konuda dikkatli olmak gerekmektedir. Çünkü bu negatif etki de edebilir. Planlama etkin şekilde 
yapılmalıdır. 

 S3 : Henüz kaybetmediğimiz fakat doğru stratejilerle kendimize bağlayabileceğimiz müşterilerdir. S4 segmentini 
tekrar kazanmaktan daha olası ve daha az maliyetlidir.Son siparişleri üzerinden geçen süre belirli bir filtreye 
sokulabilir ve filtre sonucundaki müşterilerle ilgilenilebilir. Sınırlı süreli indirimler, çapraz satış önerileri , 
yeni teklifler sunulabilir. 

"""
