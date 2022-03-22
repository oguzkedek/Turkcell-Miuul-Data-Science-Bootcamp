
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

#%%

###############################################
# Görev 1 - Veriyi Hazırlama
###############################################

# Adım 1

df_ = pd.read_excel("Hafta4/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df = df_.copy()
df.head()

"""

  Invoice StockCode                          Description Quantity         InvoiceDate  Price Customer ID        Country
0  536365    85123A   WHITE HANGING HEART T-LIGHT HOLDER        6 2010-12-01 08:26:00 2.5500  17850.0000 United Kingdom
1  536365     71053                  WHITE METAL LANTERN        6 2010-12-01 08:26:00 3.3900  17850.0000 United Kingdom
2  536365    84406B       CREAM CUPID HEARTS COAT HANGER        8 2010-12-01 08:26:00 2.7500  17850.0000 United Kingdom
3  536365    84029G  KNITTED UNION FLAG HOT WATER BOTTLE        6 2010-12-01 08:26:00 3.3900  17850.0000 United Kingdom
4  536365    84029E       RED WOOLLY HOTTIE WHITE HEART.        6 2010-12-01 08:26:00 3.3900  17850.0000 United Kingdom

"""

#%%


# Adım 2

df = df[df["StockCode"] != "POST"]
# (POST her faturaya eklenen bedel, ürünü ifade etmemektedir)

#%%

# Adım 3

(df.isnull().sum() * 100 / df.shape[0]).sort_values(ascending=False)

#Customer Id değişkeninin %24.97'si , Description değişkeninin %0.26'sı boş değerlerdem oluşmaktadır.
df.dropna(inplace=True)

df.shape
#(405633, 8)

#%%

# Adım 4

df = df[~df['Invoice'].str.contains('C', na=False)]
#. (C faturanın iptalini ifade etmektedir.)

#%%

# Adım 5

df = df[df['Price'] > 0]

# Adım 6

df.loc[:, ['Price', 'Quantity']].describe(percentiles=[.01, .1, .25, .5, .75, .95, .99])

"""
            Price    Quantity
count 396785.0000 396785.0000
mean       3.0377     13.0163
std       17.8297    179.5791
min        0.0010      1.0000
1%         0.2100      1.0000
10%        0.5500      1.0000
25%        1.2500      2.0000
50%        1.9500      6.0000
75%        3.7500     12.0000
95%        8.5000     36.0000
99%       12.7500    120.0000
max     4161.0600  80995.0000

Veri setinde aykırı değerler olduğu açıkça görülmektedir. 
"""


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, 'Quantity')
replace_with_thresholds(df, 'Price')

"""

Price sütunundaki maksimum değer 4161.06'dan 31.56'ya , Quantity sütunundaki
maksimum değer 80995'ten 298.5'e baskılanmıştır

"""

#%%

###############################################
# Görev 2 -  Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme
###############################################

df.head()

# Adım 1

# Fatura ürün pivot table'ı oluşturacak fonksiyonun yazılması .

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

#%%

# Adım 2

def create_rules(dataframe, id=True, country='Germany'):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

rules = create_rules(df)


rules[(rules['support'] > 0.05) & (rules['confidence'] > 0.05) & (rules['lift'] > 3)].  \
    sort_values('confidence', ascending=False)
"""
 - Support = X-Y'nin birlikte görülme olasılığı
 - Confidence = X satın alındığında Y'in satılması olasılığı
 - Lift = X satın alındığında Y2Nin satın alınma olasılığı lift kat kadar artar. 
 
      antecedents consequents  antecedent support  consequent support  support  confidence   lift  leverage  conviction
234      (20724)     (20719)              0.0713              0.1292   0.0601      0.8438 6.5318    0.0509      5.5733
2104     (22328)     (22326)              0.1604              0.2494   0.1336      0.8333 3.3408    0.0936      4.5033
2891     (22556)     (22554)              0.1180              0.1403   0.0690      0.5849 4.1686    0.0525      2.0711
2831     (22551)     (22554)              0.1091              0.1403   0.0624      0.5714 4.0726    0.0470      2.0059
2105     (22326)     (22328)              0.2494              0.1604   0.1336      0.5357 3.3408    0.0936      1.8085
 
"""

#%%

###############################################
# Görev 3 -  Sepet İçerisindeki Ürün Id’leri Verilen Kullanıcılara Ürün Önerisinde Bulunma
###############################################

# Adım 1

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df, 21987)
# ['PACK OF 6 SKULL PAPER CUPS']

check_id(df, 23235)
# ['STORAGE TIN VINTAGE LEAF']

check_id(df, 22747)
# ["POPPY'S PLAYHOUSE BATHROOM"]

#%%

# Adım 2

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

reco_1 = arl_recommender(rules, 21987, 1)
reco_2 = arl_recommender(rules, 23235, 1)
reco_3 = arl_recommender(rules, 22747, 1)

#%%


# Adım 3

check_id(df, reco_1[0])
#'PACK OF 6 SKULL PAPER CUPS' satın alan müşteri için önerimiz : 'PACK OF 6 SKULL PAPER PLATES'.

check_id(df, reco_2[0])
# 'STORAGE TIN VINTAGE LEAF' satın alan müşterimiz için önerimiz :'ROUND STORAGE TIN VINTAGE LEAF'

check_id(df, reco_3[0])
# "POPPY'S PLAYHOUSE BATHROOM" satın alan müşterimiz için önerimiz  "POPPY'S PLAYHOUSE LIVINGROOM "