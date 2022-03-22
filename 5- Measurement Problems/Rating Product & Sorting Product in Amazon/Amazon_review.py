###############################################
# Görev 1 - Average Rating'i Güncel Yorumlara Göre Hesaplama
###############################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv('Hafta_5/measurement_problems/datasets/amazon_review.csv')
df.head()
df.info()
df.shape
# (4915, 12)

df['overall'].mean()

# Ürünün ortalama puanı yaklaşık 4.58759'dur.

df['reviewTime'] = pd.to_datetime(df['reviewTime'])
current_date = df['reviewTime'].max()

# max tarih 2014-12-07 tarihidir.

df['days'] = (current_date - df['reviewTime']).dt.days

df['days_quantile'] = pd.qcut(df['days'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

df.loc[df['days_quantile'] == 'Q1', 'overall'].mean() * (30 / 100) + \
df.loc[df['days_quantile'] == 'Q2', 'overall'].mean() * (26 / 100) + \
df.loc[df['days_quantile'] == 'Q3', 'overall'].mean() * (24 / 100) + \
df.loc[df['days_quantile'] == 'Q4', 'overall'].mean() * (20 / 100)

# Verilen puanları tarihe göre ağırlıklı olarak hesapladık. Yeni ortalama 4.6006 .

#%%


df['day_diff_quantile'] = pd.qcut(df['day_diff'], q=4, labels=['Q1_diff', 'Q2_diff', 'Q3_diff', 'Q4_diff'])

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return df.loc[df['day_diff_quantile'] == 'Q1_diff', 'overall'].mean() * (w1 / 100) + \
           df.loc[df['day_diff_quantile'] == 'Q2_diff', 'overall'].mean() * (w2 / 100) + \
           df.loc[df['day_diff_quantile'] == 'Q3_diff', 'overall'].mean() * (w3 / 100) + \
           df.loc[df['day_diff_quantile'] == 'Q4_diff', 'overall'].mean() * (w4 / 100)

time_based_weighted_average(df)

#28,26,24,22 ağırlıklarına göre ortalama 4.5956'dır.

quantiles = ['Q1_diff', 'Q2_diff', 'Q3_diff', 'Q4_diff']

for i in quantiles :
    print(i, 'ortalaması :', df.loc[df['day_diff_quantile'] == i, 'overall'].mean())

"""
Q1_diff ortalaması : 4.6957928802588995
Q2_diff ortalaması : 4.636140637775961
Q3_diff ortalaması : 4.571661237785016
Q4_diff ortalaması : 4.4462540716612375
"""
# Ortalamalara bakıldığında ürüne yakın zamanda verilen puanların daha yüksek olduğu görülmektedir.Ürünün yükselen
#trendi olduğu söylenebilir .

#%%

###############################################
# Görev 2 - Ürün Detay Sayfasında Görülecek 20 Review
###############################################

df['helpful_no'] = df['total_vote'] - df['helpful_yes']

df["helpful_yes"].sum()
# 6444

df["helpful_no"].sum()
#1034

#%%


def score_pos_neg_diff(pos, neg):
    return pos - neg

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df['score_pos_neg_diff'] = df.apply(lambda x: score_pos_neg_diff(x['helpful_yes'],
                                                                 x['helpful_no']), axis=1)

df['score_average_rating'] = df.apply(lambda x: score_average_rating(x['helpful_yes'],
                                                                     x['helpful_no']), axis=1)

df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x['helpful_yes'],
                                                                 x['helpful_no']), axis=1)


df.sort_values('wilson_lower_bound', ascending=False).head(20)

# Wilson_lower_bound ile up oranına ait güven aralığı hesaplanır ve yanlılık problemine çözüm yöntemlerinden birisidir.
# Bernoulli dağılımından beslenen bir yapısı bulunmaktadır. Burada p başarı oranı ve 1-p başarısızlık oranı olarak
# değerlendirilir.

# Bu çalışmada up oranına ait belirlenen güven aralığına göre sıralama yapılmıştır. Sıralamanın ilk sırasında
# yer alan yorumun wlb değeri 0.95754'tür. Bu bize %95 güven aralığında (%5 yanılma payı ile) bu yoruma ait helpful_yes
# oranının alt sınırının 0.95754 olacağını gösterir.
# Sıralamadaki diğer yorumları incelediğimizde toplam yorum sayısının etkisi de açıkça görülmektedir. Örneğin
# 7 oylaması bulunan ve faydalı bulunma oranı 1 olan bir yorumun wlb değeri daha düşük seviyelerdedir. Bu bize
# bu yorumdaki toplam oy sayısının istatistiki olarak değerlendirme açısından daha zayıf kaldığını gösterir. 0.75229 faydalı
# bulunma oranı bulunan bir yorum bu yorumdan daha yüksek wlb skoruna sahiptir.