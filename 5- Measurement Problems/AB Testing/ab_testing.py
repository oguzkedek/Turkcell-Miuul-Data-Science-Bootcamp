
###############################################
# Görev 1 - Veriyi Hazırlama ve Analiz Etme
###############################################


import pandas as pd
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

kontrol = pd.read_excel('Hafta_5/measurement_problems/datasets/ab_testing.xlsx', sheet_name='Control Group')
test = pd.read_excel('Hafta_5/measurement_problems/datasets/ab_testing.xlsx', sheet_name='Test Group')

kontrol.head()
kontrol.describe([.1, .25, .5, .75, .9, .95]).T
kontrol.isnull().sum()
# Eksik değer bulunmamaktadır.

test.head()
test.describe([.1, .25, .5, .75, .9, .95]).T
test.isnull().sum()
# Eksik değer bulunmamaktadır.

# Describe ile betimsel istatistiklere bakıldığında test grubunun purchase değerlerinin daha yüksek olduğu
# görülmekte fakat bunun anlamlı bir farklılık olup olmadığının kontrolü için test yapılması gerekmektedir.

# İki farklı veri setini tek bir veri setinde birleştireceğimiz için gözlemlerin hangi veriden geldiğini
# belirtmek amacıyla yeni bir sütun oluşturukur.

kontrol['bidding'] = 'max'
test['bidding'] = 'average'

df = pd.concat([kontrol, test])

df.shape

# Veri setini alt alta ekledik ve 80 satır,5 sütunluk veri seti elde ettik.

# %%

###############################################
# Görev 2 - A/B Testinin Hipotezinin Tanımlanması
###############################################

# H0 : M1 = M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark yoktur.)
# H1 : M1!= M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark vardır.)

df.loc[df['bidding'] == 'max', 'Purchase'].mean()
df.loc[df['bidding'] == 'average', 'Purchase'].mean()

# Test grubunun ortalaması daha yüksektir fakat bunun istatistiki olarak anlamlı bir farklılık olup olmadığının kontrolü için
# A/B testi yapılacaktır.

#%%

###############################################
# Görev 3 - Hipotez Testinin Gerçekleştirilmesi
###############################################

# Normallik Varsayımı:
# H0 : Normallik varsayımı sağlanmaktadır.
# H1 : Normallik varsayımı sağlanmamaktadır.

# Kontrol grubu normallik varsayımı

test_stat, pvalue = shapiro(df.loc[df['bidding'] == 'max', 'Purchase'])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test grubu normallik varsayımı

test_stat, pvalue = shapiro(df.loc[df['bidding'] == 'average', 'Purchase'])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Kontrol grubu için p değeri 0.5891'dir.  p > 0.05 olduğu için HO reddedilemez. Dolayısıyla normallik varsayımı
# karşılanmış olur.

# Test grubu için p değeri 0.1541'dir.  p > 0.05 olduğu için HO reddedilemez. Dolayısıyla normallik varsayımı
# karşılanmış olur.

# Varyans Homojenliği Varsayımı :
# H0 : Varyanslar homojendir.
# H1 : Varyanslar homojen degildir.

test_stat, pvalue = levene(df.loc[df['bidding'] == 'max', 'Purchase'],
                           df.loc[df['bidding'] == 'average', 'Purchase'])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p değeri 0.1083'tür. Dolayısıyla HO reddedilemez .

test_stat, pvalue = ttest_ind(df.loc[df['bidding'] == 'max', 'Purchase'],
                              df.loc[df['bidding'] == 'average', 'Purchase'],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# P değeri 0.3493'tür. p > 0.05 olduğu için H0 reddedilemez.
# H0 hipotezi reddedilemediği için iki grup arasında istatistiki olarak anlamlı bir farklılık bulunmamaktadır.
# Ortalamalar arasındaki farklar tesadüftür.


###############################################
# Görev 4 - Sonuçların Analizi
###############################################

# Normallik ve homojenlik varsayımlarının yaptığımız testler sonucunda sağlandığı görülmektedir.
# Bu iki varsayımın karşılandığı koşullarda uygulayabileceğimiz bağımsız iki örneklem t testi ( parametrik test )
# bu çalışma için uygun görülmüştür.

# İşletmenin başarı ölçütü olarak belirlediği Purchase değişkeninde istatistiki olarak anlamlı bir gelişme kaydedilmediği
# görülmektedir. Bir süre daha bu değişken özelinde AB testleri yapılabilir ve gelişme kaydedilip kaydedilmediği
# gözlemlenmeye devam edilebilir.Mevcut koşullarda yeni teklif türünün daha avantajlı olduğunu bilimsel açıdan söyleyemeyiz.










