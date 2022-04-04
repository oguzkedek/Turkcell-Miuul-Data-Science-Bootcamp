import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

from helpers.data_prep import *
from helpers.eda import *

pd.set_option('display.max_columns', None)

pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

#%%

##########################################
# Görev 1 - Veriyi Hazırlama
##########################################

df = pd.read_csv('datasets/data_20K.csv')

data_understanding(df)

# Veri setinde aykırı değerlerin olduğu gözlenmektedir.
# Veri setinde boş değer bulunmamaktadır.
# Date içerikli sütunlarda veri tipi hataları bulunmaktadır, object yerine datetime ile değişmelidirler.

df.iloc[:, 3:7] = df.iloc[:, 3:7].apply(pd.to_datetime)

#cols_date = [col for col in df.columns if 'date' in col]
#df[cols_date] = df[cols_date].apply(pd.to_datetime)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Veri setinde 3 cat 8 num ve 2 cat_but_car sütun bulunmaktadır.

for col in cat_cols:
    cat_summary(df, col)

plt.figure(figsize=(15, 15))
for i, col in enumerate(cat_cols):
    plt.subplot(3, 1, i+1)
    ax = sns.countplot(x=col, data=df)
    plt.title(f"Count Analysis of {col} Features", fontsize=12)
plt.show()

# order_channel sütununa baktığımızda veri seti içerisinde Android App %47.60 , Mobile %24.47, Ios App %14.20 ve
#Desktop %13.71 oranlarında bulunmaktadır.
# last_order_channel değişkeninde de en yüksek bulunma oranına sahip değer %34.00 oranıyla Android App'tir. Offline ise
# bu kırılımda %33.13 ile ikinci en çok bulunma yüzdesine sahip değerdir.
# Store_type sütununda yalnızca A tipi storeun % 77.47 oranıyla üstünlüğü bulunmaktadır.

#%%

num_cols = [col for col in num_cols if 'date' not in col]

for col in num_cols:
    num_summary(df, col)

plt.figure(figsize=(15, 25))
for i, col in enumerate(num_cols):
    plt.subplot(2, 2, i+1)
    ax = sns.histplot(data=df, x=col, kde=True)
    plt.title(f"Distribution Analysis of {col} Features", fontsize=12)
plt.show()

# Adım 2 -

df["interested_in_categories_12"].unique()

df['interested_in_categories_12'] = df['interested_in_categories_12'].str.replace("[", "")
df['interested_in_categories_12'] = df['interested_in_categories_12'].str.replace("]", "")
df['interested_in_categories_12'] = df['interested_in_categories_12'].str.replace("'", "")
one_hot = MultiLabelBinarizer()
one_hot.classes_
y_classes = one_hot.fit_transform(df['interested_in_categories_12'].str.split(', '))
y_classes = pd.DataFrame(y_classes, columns=one_hot.classes_)
y_classes = pd.DataFrame(y_classes,
                         columns=['uncategorized', 'aktif_cocuk', 'aktif_spor', 'cocuk', 'erkek', 'kadin'])

#       uncategorized  aktif_cocuk  aktif_spor  cocuk  erkek  kadin
#0                  0            0           0      0      0      1
#1                  0            0           1      1      1      1
#2                  0            0           0      0      1      1
#3                  0            1           0      1      0      0
#4                  0            0           1      0      0      0

df = pd.concat([df, y_classes], axis=1)

df['frequency'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['monetary'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']

# en son kaç gün önce alışveriş yapıldı ?
today_date = df['last_order_date'].max() + pd.DateOffset(days=2)
df['recency'] = (today_date - df['last_order_date']).dt.days

#müşterinin yaşı
df['tenure'] = (today_date - df["first_order_date"]).dt.days

df_ = df.copy()
df_hier = df_.copy()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if 'date' not in col]

for col in num_cols:
    print(col, check_outlier(df, col))
for col in num_cols:
    replace_with_thresholds(df, col)

#%%

##############################################
# Görev 2 - K-Means ile Müşteri Segmentasyonu
##############################################

# Adım 1 -

df.head()

sc = MinMaxScaler((0, 1))
df[num_cols] = sc.fit_transform(df[num_cols])

# Adım 2 -
date_cols = [col for col in df.columns if 'date' in col]
df.drop(date_cols, axis=1, inplace=True)
useless_cols = ['interested_in_categories_12', 'master_id']
df.drop(useless_cols, axis=1, inplace=True)
df = one_hot_encoder(df, cat_cols)

df.head()

kmeans = KMeans()
ssd = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

#%%

# Silüet Yöntemi

sil = []
K_sil = list(range(2, 10))

for k in K_sil:
  kmeans = KMeans(n_clusters=k).fit(df)
  labels = kmeans.labels_
  sil.append(silhouette_score(df, labels, metric='euclidean'))

plt.plot(K_sil, sil, linestyle='--', marker='o', color='r');
plt.xlabel('K')
plt.ylabel('Silhouette score')
plt.title('Silhouette score vs. K')
plt.show()

# Silhouette skora göre 9 optimum küme sayısını verirken dirsek yöntemine göre 7 vermektedir.

#%%

# Adım 3

kmeans = KMeans(n_clusters=7, random_state=42).fit(df)

kmeans.cluster_centers_
kmeans.labels_

clusters_kmeans = kmeans.labels_
df_["cluster"] = clusters_kmeans
df_.head(10)

#%%

# Adım 4-

df_['cluster'].value_counts()
df_.groupby('cluster')[['frequency', 'monetary', 'recency', 'tenure']]. \
    agg(['min', 'max', 'mean', 'median'])

"""
- Cluster0 en yüksek monetary medyanına sahip segmenttir. Aynu zamanda tenure açısından bakıldığında en 
yüksek müşteri yaşı medyanına sahip segmenttir.
-Cluster4 recency açısından en başarılı segmenttir. Tenure açısından ise 2.sırada olan segmenttir.
-Cluster1 en düşük monetary medyanına sahip segmenttir.

"""

#%%

####################################################################
# Görev 3 - Hierarchical Clustering ile Müşteri Segmentasyonu
####################################################################

# Adım 1 -

df.describe().T

hc_average = linkage(df, "complete")

plt.figure(figsize=(20, 15))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dend = dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=3.25, color='r', linestyle='--')
plt.axhline(y=3.375, color='b', linestyle='--')
plt.axhline(y=3.6, color='g', linestyle='--')
plt.show()

# Küme sayısı 8 olarak belirlenmiştir.

#%%

# Adım 2-

cluster = AgglomerativeClustering(n_clusters=8, linkage="complete")

clusters = cluster.fit_predict(df)

df_hier["hi_cluster_no"] = clusters
df_hier["kmeans_cluster_no"] = clusters_kmeans

df_hier.head()

#%%

# Adım 3-

df_hier['hi_cluster_no'].value_counts()
df_hier.groupby('hi_cluster_no')[['frequency', 'monetary', 'recency', 'tenure']].agg(['min', 'max', 'mean', 'median'])

"""

- Cluster 2 en yüksek monetary medyanına sahip segmenttir. Aynı zamanda recency değeri en düşük ve tenure değeri en
 yüksek olan segmenttir .
- Cluster 1 en düşük frekans medyanına sahip segmentlerdir. Cluster 1 aynı zamanda en düşük monetary
ortalamasına sahip segment olmasına karşın recency değeri en düşük 2.segmenttir. 

"""