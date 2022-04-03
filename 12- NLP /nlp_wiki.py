
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from wordcloud import WordCloud, ImageColorGenerator
import nltk

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv('datasets/wiki_data.csv', index_col=0)
df.head(10)

#########################################################
# Görev 1:  Metin Ön İşleme İşlemlerini Gerçekleştiriniz
#########################################################

# Adım 1

def clean_text(dframe, col):
    dframe[col] = dframe[col].str.lower()
    dframe[col] = dframe[col].str.replace('[^\w\s]', '')
    dframe[col] = dframe[col].str.replace('\d', '')


# Adım 2

clean_text(df, 'text')
df.head(10)

# Adım 3

def remove_stopwords(dframe, col, language='english'):
    sw = stopwords.words(language)
    dframe[col] = dframe[col].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# Adım 4

remove_stopwords(df, 'text')

# Adım 5

temp_df = pd.Series(' '.join(df['text']).split()).value_counts()
drops = temp_df[temp_df <= 1000]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# Adım 6

df["text"].apply(lambda x: TextBlob(x).words).head()

"""
Out[54]: 
1    [computer, services, company, based, france, f...
2    [battery, battery, also, known, battery, devic...
"""

# Adım 7

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


#########################################################
# Görev 2:  Veriyi Görselleştiriniz .
#########################################################

df['text'] = df['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in ' â '))

# Adım 1


tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)[:20]

# Adım 2

tf[tf["tf"] > 8000].plot.bar(x="words", y="tf")
plt.show()

# Adım 3

text = " ".join(i for i in df.text)
mask = np.array(Image.open("Hafta_12/wiki_.jpg"))
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="black",
                      mask=mask).generate(text)
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.show()

#########################################################
# Görev 3:  Tüm Aşamaları Tek Bir Fonksiyon Olarak Yazınız
#########################################################

def nlp_text(dframe, col, language='english', threshold_drop=1000, bar_plot=False, threshold_bar=8000, word_cloud=False):
    """
    Veri setinde kullanıcının belirlediği ve text içeren kolon üzerinde metin ön işleme adımları uygulayan ve görselleştiren
    fonksiyondur.

    Parameters
    ------------

        dframe : dataframe
                Üzerinde işlem yapılacak dataframe.

        col :   string
                Üzerinde işlem yapılacak sütun.

        language : string, optional
                Stop words işlemi yapılacak metinin dili.

        threshold_drop: int, optional
                Metinde az geçen kelimeler için belirlenen eşik değeri.

        bar_plot: bool, optional
                Metindeki terimlerin frekanslarını bar plot grafik ile görselleştirir.

        threshold_bar: int, optional
                Bar plot kullanılması durumunda grafikte gösterilecek terimlerin frekansının eşik değeri.

        word_cloud : bool, optional
                Metindeki terimlerin frekanslarını word cloud ile görselleştirir.

    Examples
    ------------

    df dataframe'inde yer alan text sütunundaki terimlerin frekanslarını word cloud ile görselleştiren fonksiyon.

    df = pd.read_csv('datasets/wiki_data.csv', index_col=0)
    nlp_text(df, 'text', word_cloud=True)

    Returns
    ------------

    dataframe : Metin ön işleme işlemlerinin gerçekleştirildiği dataframe.

    """
    dframe[col] = dframe[col].str.lower()
    dframe[col] = dframe[col].str.replace('[^\w\s]', '')
    dframe[col] = dframe[col].str.replace('\d', '')

    sw = stopwords.words(language)
    dframe[col] = dframe[col].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

    temp_df = pd.Series(' '.join(df['text']).split()).value_counts()
    drops = temp_df[temp_df <= threshold_drop]
    dframe[col] = dframe[col].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

    dframe[col] = dframe[col].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    dframe[col] = dframe[col].apply(lambda x: " ".join(x for x in str(x).split() if x not in ' â '))

    if bar_plot:

        tf = dframe[col].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        tf.columns = ["words", "tf"]
        tf[tf["tf"] > threshold_bar].plot.bar(x="words", y="tf")
        plt.show()

    if word_cloud:

        text = " ".join(i for i in dframe[col])
        mask = np.array(Image.open("Hafta_12/wiki_.jpg"))
        wordcloud = WordCloud(max_font_size=50,
                              max_words=100,
                              background_color="black",
                              mask=mask).generate(text)
        image_colors = ImageColorGenerator(mask)
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
        plt.axis("off")
        plt.show()

nlp_text(df, 'text', word_cloud=True)