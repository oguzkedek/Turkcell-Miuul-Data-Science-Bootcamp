###############################################
# ÖDEV 1: List Comprehension Applications
###############################################

###############################################
# Görev 1: car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.
###############################################

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = sns.load_dataset("car_crashes")

# Veri setini baştan okutarak aşağıdaki çıktıyı elde etmeye çalışınız.

# ['NUM_TOTAL',
#  'NUM_SPEEDING',
#  'NUM_ALCOHOL',
#  'NUM_NOT_DISTRACTED',
#  'NUM_NO_PREVIOUS',
#  'NUM_INS_PREMIUM',
#  'NUM_INS_LOSSES',
#  'ABBREV']

# Notlar:
# Numerik olmayanların da isimleri büyümeli.
# Tek bir list comp yapısı ile yapılmalı.

df.info()

###############################################
# Görev 1 Çözüm

new_cols = ['NUM_' + col.upper() if df[col].dtype != 'O' else col.upper() for col in df.columns]

###############################################


###############################################
# Görev 2: İsminde "no" BARINDIRMAYAN değişkenlerin isimlerininin SONUNA "FLAG" yazınız.
###############################################

# Tüm değişken isimleri büyük olmalı.
# Tek bir list comp ile yapılmalı.

# Beklenen çıktı:

# ['TOTAL_FLAG',
#  'SPEEDING_FLAG',
#  'ALCOHOL_FLAG',
#  'NOT_DISTRACTED',
#  'NO_PREVIOUS',
#  'INS_PREMIUM_FLAG',
#  'INS_LOSSES_FLAG',
#  'ABBREV_FLAG']

###############################################
# Görev 2 Çözüm

new_cols = [col.upper() + '_FLAG' if 'no' not in col else col.upper() for col in df.columns]

###############################################

###############################################
# Görev 3: Aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçerek yeni bir df oluşturunuz.
###############################################

# df.columns
# og_list = ["abbrev", "no_previous"]

# Önce yukarıdaki listeye göre list comprehension kullanarak new_cols adında yeni liste oluşturunuz.
# Sonra df[new_cols] ile bu değişkenleri seçerek yeni bir df oluşturunuz adını new_df olarak isimlendiriniz.


# Beklenen çıktı:

# new_df.head()
#
#    total  speeding  alcohol  not_distracted  ins_premium  ins_losses
# 0 18.800     7.332    5.640          18.048      784.550     145.080
# 1 18.100     7.421    4.525          16.290     1053.480     133.930
# 2 18.600     6.510    5.208          15.624      899.470     110.350
# 3 22.400     4.032    5.824          21.056      827.340     142.390
# 4 12.000     4.200    3.360          10.920      878.410     165.630

###############################################
# Görev 3 Çözüm

og_list = ["abbrev", "no_previous"]
new_cols = [col for col in df.columns if col not in og_list]
new_df = df[new_cols]
new_df.head()



###############################################
# ÖDEV 2: Fonksiyonlara Özellik Eklemek.
###############################################

# ÖNCESİ

def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

#SONRASI

def cat_summary(dataframe, col_name, target, plot=False, bar=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("="*50)
    if bar:
        sns.barplot(x=dataframe[col_name], y=dataframe[target], data=dataframe)
        plt.show()
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

df = pd.read_csv("datasets/titanic.csv")
cat_summary(df, "Embarked", 'Survived', bar=True)

###############################################
# ÖDEV 3: Docstring.
###############################################
# Aşağıdaki fonksiyona 4 bilgi (uygunsa) barındıran numpy tarzı docstring yazınız.
# (task, params, return, example)
# cat_summary()

def cat_summary(dataframe, col_name, target,plot=False,bar=False):
    """
    Print the counts and ratio of unique values in the given column.
    Show point estimates and confidence intervals as rectangular bars for the given column.
    Show the counts of observations in each categorical bin using bars
    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        col_name: str
                The column with the categorical variables to be analyzed
        target : str
                The target variable to be used in bar plot analysis
        plot: bool, optional
                Plotting of the categorical variables
        bar : bool,optional
                Plotting of the categorical variables
    Examples
    ------
    import seaborn as sns
    import pandas as pd
    df = pd.read_csv("datasets/titanic.csv")
    cat_summary(df, "Embarked", 'Survived', bar=True)

   """
help(cat_summary)