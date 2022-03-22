#######################
# Oğuz Kedek - FLO Case - Endüstri Projesi 1
#######################

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


###############################################
# Görev 1 Veriyi Anlama ve Hazırlama
###############################################

df_ = pd.read_csv('Hafta_3/FLO_RFM_Analizi/flo_data_20K.csv')
df = df_.copy()

df.head(10)  # İlk 10 gözlem

df.columns   #Değişken isimleri

df.describe(percentiles=[.01, .10, .25, .5, .75, .9, .99]).T

# Betimsel İstatistik (daha fazla quartile açısından gözlemlemek daha faydalı olabilir)
# Aykırı değerler olduğu açıkça gözlemlenmektedir fakat bu çalışma kapsamında ilgilenilmeyecektir.

df.isnull().sum()
# Boş değer analizi - Veri setinde boş değer bulunmamaktadır.

df.info()
#df.dtypes - Date değişkenlerinin veri tipi değiştirilmelidir .

#%%

df['omnichannel_order_num_total'] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df['omnichannel_customer_value_total'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']
#Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturulmuştur.

df.info()
df.iloc[:, 3:7] = df.iloc[:, 3:7].apply(pd.to_datetime)
# hangi sütunlarda date ifadesinin yer aldığının kolayca gözlemlenebildiği durum

#cols_date = [col for col in df.columns if 'date' in col]
#fazla veya karmaşık sıralarda sütun olduğu durum (fonksiyonlaştırmada bu yöntem kullanılacaktır)
#df[cols_date] = df[cols_date].apply(pd.to_datetime)

channel_results = df.groupby('order_channel').agg({'master_id': 'count',
                                                   'omnichannel_order_num_total': 'mean',
                                                   'omnichannel_customer_value_total': 'mean'})

channel_results.columns = ['unique_customer', 'omnichannel_order_mean', 'omnichannel_customer_value_mean']

"""

- Android App kanalındaki müşteri sayısının diğer kanallara göre oldukça fazla olduğu gözlenmektedir.
- Offline ve online harcamaların toplamını alarak oluşturduğumuz omnichannel_customer_value ortalamasına baktığımızda
Ios App kanalı en fazla harcama ortalamasına sahip kanaldır.
- Desktop kanalı toplam yapılan harcama ortalaması bakımından en düşük seviyedeki kanaldır.

"""
# %%

df.sort_values(by='omnichannel_customer_value_total', ascending=False).head(10)
# En fazla kazancı getiren 10 müşteri .

df.sort_values(by='omnichannel_order_num_total', ascending=False).head(10)
# En fazla siparişi veren 10 müşteri .

# En fazla kazanç sağlayan müşteriler ile en fazla sipariş veren müşteriler arasında farklılıklar olduğu gözlemlenmektedir.
# Burada alınan ürünün fiyatının da önemi bulunmaktadır. Örneğin 4 sipariş yapmış bir müşteri en fazla kazanç sağlayanlar
# arasında 5.sırada olabilmektedir.

#%%

# ÇALIŞMAYI FONKSİYONLAŞTIRMA

def data_prepare(df, percentiles=[.01,.10,.25,.5,.75,.9,.99]):
    #Adım 2
    print('\033[1m', 'İlk 10 Gözlem', '\033[0m', df.head(10), sep='\n')
    print('-'*150)
    print('\033[1m', 'Değişken İsimleri', '\033[0m', df.columns, sep='\n')
    print('-' * 150)
    print('\033[1m', 'Betimsel İstatistik', '\033[0m', df.describe(percentiles=percentiles).T, sep='\n')
    print('-' * 150)
    print('\033[1m', 'Boş Değer Check', '\033[0m', df.isnull().sum(), sep='\n')
    print('-' * 150)
    print('\033[1m', 'Değişken Tipleri İncelemesi', '\033[0m', df.dtypes, sep='\n')
    print('-' * 150)

    #Adım 3
    df['omnichannel_order_num_total'] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df['omnichannel_customer_value_total'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']

    #Adım 4
    cols_date = [col for col in df.columns if "date" in col]
    df[cols_date] = df[cols_date].apply(pd.to_datetime)

    #Adım 5
    channel_results = df.groupby('order_channel').agg({'master_id': 'count',
                                                       'omnichannel_order_num_total': 'mean',
                                                       'omnichannel_customer_value_total': 'mean'})
    channel_results.columns = ['unique_customer', 'omnichannel_order_mean', 'omnichannel_customer_value_mean']
    print('\033[1m', 'Alışveriş Kanalları Analizi', '\033[0m', channel_results, sep='\n')
    print('-' * 150)

    #Adım 6
    print('\033[1m', 'En Fazla Kazanç Getiren 10 Müşterinin Analizi', '\033[0m',
          df.sort_values(by='omnichannel_customer_value_total', ascending=False).head(10), sep='\n')
    print('-' * 150)

    #Adım 7
    print('\033[1m', 'En Fazla Sipariş Veren 10 Müşterinin Analizi', '\033[0m',
          df.sort_values(by='omnichannel_order_num_total', ascending=False).head(10), sep='\n')

data_prepare(df)

#%%


###############################################################
# Görev 2: RFM Metriklerinin Hesaplanması
###############################################################

"""
- Recency   : Son alışverişten sonra geçen süredir .
Recency'nin düşük olması bize müşterinin yakın zamanda alış yaptığını gösterir.

- Frequency : Toplam alışveriş sayısı, sıklığıdır.
Müşterilerin frequency değerlerinin yüksek olmasını isteriz.

- Monetary  : Tüm alışverişlerde harcanan toplam paradır. İşletmeler bu değerin yüksek olmasını beklemektedir. 
"""

today_date = df['last_order_date'].max() + pd.DateOffset(days=2)

df['recency'] = (today_date - df['last_order_date']).dt.days

rfm = df[['recency', 'omnichannel_order_num_total', 'omnichannel_customer_value_total']]

#rfm.rename(columns={'omnichannel_order_num_total': 'frequency', 'omnichannel_customer_value_total': 'monetary'}, inplace=True)

rfm.columns = ['recency', 'frequency', 'monetary']
rfm.index = df['master_id']

#%%

###############################################################
# Görev 3: RF Skorunun Hesaplanması
###############################################################

rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm['monetary_score'] = pd.qcut(rfm['recency'], 5, labels=[1, 2, 3, 4, 5])

rfm['RF_SCORE'] = (rfm['recency_score'].astype(str) +
                   rfm['frequency_score'].astype(str))

rfm.head(10)

#%%

###############################################################
# Görev 4: RF Skorunun Segment Olarak Tanımlanması
###############################################################

"""
Not : Recency değerleri skor olarak değil ilk değeri açısından yorumlanmıştır. 

- Hibernating : İnaktif durumunda olan , işletme için en faydasız konumdaki müşterilerdir. Recency değerleri en yüksek ve 
frequency değerleri en düşük olan müşterilerdir. 

- At Risk : Hibernating müşterilere göre nispeten daha sık alışveriş yapmış olmalarına karşın recency değerleri aynı şekilde yüksek 
olan müşterileridir 

- Can't loose : Frequency değerleri açısından işletmenin kaybetmemesi gereken ve üzerinde aksiyonlar alınması gereken müşterilerdir
Son alımları eski bir tarih olmasına rağmen değerli müşterilerdir. 

- About to Sleep : Recency değerleri ortalama seviyededir. Alışveriş sıklığı en alt seviyededir .

- Need Attention : Ortalama recency ve frequency değerlerine sahip müşterilerdir. Dikkat edilmemesi durumunda hibernating,
at_risk,about_to_sleep gibi segmentlerine düşmeleri muhtemeldir . Bu müşterilerin seviyesinin yukarı çekilmesi için 
aksiyon alınması gereklidir.

- Loyal_customers : İşletmenin sadık müşterileridir. Orta-yüksek recency değerine ve yüksek frequency değerlerine sahiptirler.

- Promising : Yakın zamanda işletmeden alışveriş yapmış müşterilerdir fakat sıklıkları düşük seviyededir. 

- New Customers : En düşük recency değerine (en yüksek recency skoru) sahip müşterilerdir . 

- Potential_loyalist : Yakın zamanda alışveriş yapmış ve frequency değerleri orta - düşük seviyede olan müşterilerdir. 
Frequency değerlerinin artması durumunda sadık müşteriler hatta champions grubuna dahi girebilirler.

- Champions : Hem en yakın zamanda hem de en sık alışveriş yapan müşterilerdir. 

"""

seg_map = {
    r'[1-2]{2}': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

#%%
rfm.loc[:, rfm.columns !='RF_SCORE'].groupby('segment').agg(['mean', 'count'])

"""
- champions ve new customers segmentlerinin yakın zamanlarda alışveriş yaptığı görülmektedir.
- Burada ayırt edici nokta işlem sıklığıdır. Aynı zamanda champions segmentinin monetary değerleri de
tatmin edici seviyededir.
-can't lose segmenti yakın zamanda alışveriş yapmamasına rağmen en sık alışveriş yapan segment konumundadır
ve bizim için değerlidir. İşletmeye en çok para kazandıran segmenttir.
 
"""

rfm[rfm["segment"] == "cant_loose"].head()

df['interested_in_categories_12'].unique() # Unique kategorilerin incelenmesi.

df_all = df.merge(rfm, how='inner', on=('master_id', 'recency'))
#İki dataframe'de de recency sütunu bulunduğu için on parametresine recency de eklenmiştir ve suffixes durumu kurtarılmıştır.

case_a = df_all.loc[((df_all['segment'] == 'champions') | (df_all['segment'] == 'loyal_customers')) &
                    (df_all['monetary'] / df_all['frequency'] > 250) &
                    (df_all['interested_in_categories_12'].str.contains('KADIN'))].master_id

case_a.to_csv("customers.csv", index=False)