-- 1 / Customers isimli bir veritabanı oluşturunuz. 

CREATE DATABASE Customers

SELECT * FROM CUSTOMERS

-- Müşteri tablosu import edilmiştir. 

--2 / Toplam yapılan ciroyu getirecek sorguyu yazınız.

SELECT SUM(customer_value_total_ever_offline + customer_value_total_ever_online) CIRO 
FROM CUSTOMERS

/*
     CIRO 
     14983567.310000971

*/

--3 / Fatura başına yapılan ortalama ciroyu getirecek sorguyu yazınız.

SELECT  (SUM(customer_value_total_ever_offline + customer_value_total_ever_online) / SUM(order_num_total_ever_online + order_num_total_ever_offline)) ORT_CIRO
FROM CUSTOMERS

/*
    ORT_CIRO 
    149.50825003243867

*/

--4 / Son alışveriş platformları üzerinden yapılan alışverişlerin toplam ciro dağılımlarını getirecek sorguyu yazınız.


SELECT last_order_channel SON_KANAL , SUM(customer_value_total_ever_online + customer_value_total_ever_offline) TOPLAM_CIRO
FROM CUSTOMERS
GROUP BY last_order_channel

/*
    SON_KANAL   TOPLAM_CIRO     
    Offline	    4435677.8100000685
    Desktop	    1475927.6499999904
    Android App	5618108.890000089
    Mobile	    2038731.5399999616
    Ios App	    1415121.4199999862

Not : Case when ile offline-online ayrımı da yapılabilir. 

*/

--5 / Toplam yapılan fatura adedini getirecek sorguyu yazınız.

SELECT SUM(order_num_total_ever_online + order_num_total_ever_offline) FATURA_SAYISI
FROM CUSTOMERS

/*

Alışveriş sayısı kadar fatura kesildiği varsayılmıştır. 

    FATURA_SAYISI
    100219
*/

--6 / Alışveriş yapanların son alışveriş yaptıkları platform dağılımlarını fatura cinsinden getirecek sorguyu yazınız.

SELECT last_order_channel SON_KANAL , SUM(order_num_total_ever_online + order_num_total_ever_offline) FATURA_SAYISI
FROM CUSTOMERS
GROUP BY last_order_channel

/*
    SON_KANAL FATURA_SAYISI
    Offline	    30643
    Desktop	    9466
    Android App	37320
    Mobile	    14195
    Ios App	    8595
*/

--7 / Toplam yapılan ürün satış miktarını getirecek sorguyuyazınız.

SELECT SUM(order_num_total_ever_online + order_num_total_ever_offline) ÜRÜN_SATIS
FROM CUSTOMERS

/*

Ürün satışı ile ilgili bu bilgiye sahip değiliz fakat alışveriş sayısı bilgisi verilebilir. 

    ÜRÜN_SATIS
    100219

*/

--8 / Yıl kırılımında ürün adetlerini getirecek sorguyu yazınız .

-- Unique faturalara sahip değiliz. O yüzden yıl kırılımında doğru sonuca ulaşamayız. İki farklı sene arasındaki alışverişleri değerlendireceğim.


SELECT 
DATEPART(YEAR,first_order_date) ILK_ALISVERIS_YILI, DATEPART(YEAR,last_order_date) SON_ALISVERIS_YILI, 
SUM(order_num_total_ever_offline + order_num_total_ever_online) ALISVERIS_SAYISI
FROM CUSTOMERS
GROUP BY DATEPART(YEAR,first_order_date), DATEPART(YEAR,last_order_date)
ORDER BY 2,1

/*
    ILK_ALISVERIS_YILI      SON_ALISVERIS_YILI     ALISVERIS_SAYISI
    2013	                    2020	            673
    2014	                    2020	            1421
    2015	                    2020	            1836
    2016	                    2020	            2043
    2017	                    2020	            3339
*/

-- 9 / Platform kırılımında ürün adedi ortalamasını getirecek sorguyu yazınız. 

SELECT order_channel PLATFORM , AVG(order_num_total_ever_offline + order_num_total_ever_online) ORTALAMA_ALISVERIS
FROM CUSTOMERS
GROUP BY order_channel


/*
    PLATFORM        ORTALAMA_ALISVERIS
    Desktop	        3.992687
    Android App	    5.504897
    Mobile	        4.440598
    Ios App	        5.418637
*/

SELECT AVG(order_num_total_ever_offline) OFFLINE_ORT, AVG(order_num_total_ever_online) ONLINE_ORT
FROM CUSTOMERS

/*
    OFFLINE_ORT     ONLINE_ORT
    1.913913	    3.110854
*/

-- 10 / Kaç adet farklı kişinin alışveriş yaptığını gösterecek sorguyu yazınız.

SELECT COUNT(DISTINCT master_id) UNIQUE_MUSTERI_SAYISI 
FROM CUSTOMERS

/*
    UNIQUE_MUSTERI_SAYISI
    19945
*/

--11 / Son 12 ayda en çok ilgi gören kategoriyi getiren sorguyu yazınız.

SELECT TOP 1 interested_in_categories_12 KATEGORİ ,COUNT(interested_in_categories_12) MUSTERI_SAYISI 
FROM CUSTOMERS
GROUP BY interested_in_categories_12
ORDER BY 2 DESC

/*

    KATEGORİ        MUSTERI_SAYISI
    [AKTIFSPOR]	    3464
*/

--12 / Kanal kırılımında en çok ilgi gören kategorileri getiren sorguyu yazınız.

SELECT order_channel KANAL ,interested_in_categories_12 KATEGORİ, COUNT(interested_in_categories_12) MUSTERI_SAYISI 
FROM CUSTOMERS
GROUP BY order_channel , interested_in_categories_12
ORDER BY 1 ,3 DESC

/*
    KANAL           KATEGORİ        MUSTERI_SAYISI  
    Android App	    [AKTIFSPOR]	    1570
    Android App	    [ERKEK]	        938
    Android App	    []	            933
    Android App	    [KADIN]	        803
    .
    .
    .
*/

--13 / En çok tercih edilen store type’ları getiren sorguyu yazınız. 

SELECT TOP 1 store_type MAGAZA_TİPİ, COUNT(store_type) MUSTERİ_SAYISI 
FROM CUSTOMERS
GROUP BY store_type
ORDER BY 2 DESC

/*
    MAGAZA_TİPİ     MUSTERI_SAYISI  
    A	            15453
*/

--14 / Store type kırılımında elde edilen toplam ciroyu getiren sorguyu yazınız.

SELECT  store_type MAGAZA_TİPİ,  SUM(customer_value_total_ever_online + customer_value_total_ever_offline) TOPLAM_CIRO
FROM CUSTOMERS
GROUP BY store_type

/*
    MAGAZA_TIPI     TOPLAM_CIRO
    A,B,C	        92189.99
    A	            10840131.790000452
    A,B	            3922809.7100000298
    A,C	            128435.81999999998
*/

--15 / Kanal kırılımında en çok ilgi gören store type’ı getiren sorguyu yazınız.

SELECT TOP 4 order_channel KANAL , store_type MAGAZA_TIPI, COUNT(store_type) MUSTERİ_SAYISI
FROM CUSTOMERS
GROUP BY order_channel, store_type
ORDER BY 3 DESC

/*  
    KANAL       MAGAZA_TIPI    MUSTERI_SAYISI
    Android App	    A	        7573
    Mobile	        A	        3849
    Desktop	        A	        2093
    Ios App	        A	        1938
*/

-- 16 / En çok alışveriş yapan kişinin ID’sini getiren sorguyu yazınız.

SELECT TOP 1 master_id MUSTERI_ID , (order_num_total_ever_online + order_num_total_ever_offline) FATURA_SAYISI 
FROM CUSTOMERS
ORDER BY 2 DESC


/*
    MUSTER_ID                              FATURA_SAYISI
    5d1c466a-9cfd-11e9-9897-000d3a38a36f	202
*/

-- 17 / En çok alışveriş yapan kişinin fatura başı ortalamasını getiren sOrguyu yazınız.

SELECT TOP 1 master_id MUSTERI_ID , (order_num_total_ever_online + order_num_total_ever_offline) FATURA_SAYISI , 
((customer_value_total_ever_online + customer_value_total_ever_offline) / 
(order_num_total_ever_online + order_num_total_ever_offline)) ORT_HARCAMA
FROM CUSTOMERS
ORDER BY 2 DESC

/*
    En çok alışveriş yapan kişinin id'sini bilmediğimiz varsayılmıştır.

    MUSTERI_ID                              FATURA_SAYISI   ORT_HARCAMA                            
    5d1c466a-9cfd-11e9-9897-000d3a38a36f	202	            227.2529702970297
*/

--18 / En çok alışveriş yapan kişinin alışveriş yapma gün ortalamasını getiren sorguyu yazınız.

SELECT master_id,DATEDIFF(DAY,first_order_date, last_order_date) GUN_FARKI, 
DATEDIFF(DAY,first_order_date, last_order_date) / (order_num_total_ever_offline + order_num_total_ever_online) ALISVERIS_SIKLIGI 
FROM CUSTOMERS
WHERE master_id = '5d1c466a-9cfd-11e9-9897-000d3a38a36f'

/*
    En çok alışveriş yapan kişinin id'sini bildiğimiz varsayılmıştır.

    master_id	                            GUN_FARKI	ALISVERIS_SIKLIGI
    5d1c466a-9cfd-11e9-9897-000d3a38a36f	2758	    13.65346534653465346534
*/

-- 19 / En çok alışveriş yapan ilk 100 kişinin (ciro bazında) alışveriş yapma gün ortalamasını getiren sorguyu yazınız.

SELECT TOP 100 master_id MUSTERI_ID,
(customer_value_total_ever_online + customer_value_total_ever_offline) TOPLAM_CIRO,
(DATEDIFF(DAY,first_order_date, last_order_date)) / (order_num_total_ever_offline + order_num_total_ever_online) ALISVERIS_SIKLIGI
FROM CUSTOMERS
ORDER BY 2 DESC

/*
    MUSTERI_ID                              TOPLAM_CIRO     ALISVERIS_SIKLIGI
    5d1c466a-9cfd-11e9-9897-000d3a38a36f	45905.1	        13.65346534653465346534
    d5ef8058-a5c6-11e9-a2fc-000d3a38a36f	36818.29	    13.70588235294117647058
    73fd19aa-9e37-11e9-9897-000d3a38a36f	33918.1	        32.69512195121951219512
    7137a5c0-7aad-11ea-8f20-000d3a38a36f	31227.41	    3.90909090909090909090
    47a642fe-975b-11eb-8c2a-000d3a38a36f	20706.34	    5.00000000000000000000
    a4d534a2-5b1b-11eb-8dbd-000d3a38a36f	18443.57	    6.27142857142857142857
    d696c654-2633-11ea-8e1c-000d3a38a36f	16918.57	    20.48571428571428571428
    fef57ffa-aae6-11e9-a2fc-000d3a38a36f	12726.1	        44.48648648648648648648
    .
    .
    .
*/

-- 20 / Platfrom kırılımında en çok alışveriş yapan müşteriyi getiren sorguyu yazınız.




--21 /  En son alışveriş yapan kişinin ID’sini getiren sorguyu yazınız. (Max son tarihte birden fazla alışveriş yapan ID bulunmakta. Bunları da getiriniz.)

SELECT master_id MUSTERI_ID, last_order_date TARIH
FROM CUSTOMERS
WHERE last_order_date = (SELECT MAX(last_order_date) FROM CUSTOMERS)

/*
    MUSTERI_ID                              TARIH
    241f0ad0-afb5-11e9-9757-000d3a38a36f	2021-05-30
    9613613c-c9d0-11ea-a31e-000d3a38a36f	2021-05-30
    d4fdf864-2852-11ea-b87a-000d3a38a36f	2021-05-30
    62b2a42a-2a74-11ea-b3a7-000d3a38a36f	2021-05-30
    .
    .
    .
*/

-- 22 / En son alışveriş yapan kişinin alışveriş yapma gün ortalamasını getiren sorguyu yazınız.

SELECT TOP 1 master_id MUSTERI_ID, last_order_date TARIH,
(DATEDIFF(DAY,first_order_date, last_order_date)) / (order_num_total_ever_offline + order_num_total_ever_online) ALISVERIS_SIKLIGI
FROM CUSTOMERS
WHERE last_order_date = (SELECT MAX(last_order_date) FROM CUSTOMERS)

/*
    MUSTERI_ID	                            TARIH	          ALISVERIS_SIKLIGI
    241f0ad0-afb5-11e9-9757-000d3a38a36f	2021-05-30	36.83333333333333333333
*/

-- 23 /  Platform kırılımında en son alışveriş yapan kişilerin fatura başına ortalamasını getiren sorguyu yazınız.



-- 24 /  İlk alışverişini yapan kişinin ID’sini getiren sorguyu yazınız.

SELECT master_id MUSTERI_ID, first_order_date TARIH
FROM CUSTOMERS
WHERE first_order_date = (SELECT MIN(first_order_date) FROM CUSTOMERS)

/*
    MUSTERI_ID	                                TARIH
    1d033d3a-b090-11e9-9757-000d3a38a36f	2013-01-14
*/


-- 25 /  İlk alışveriş yapan kişinin alışveriş yapma gün ortalamasını getiren sorguyu yazınız.

SELECT TOP 1 master_id MUSTERI_ID, first_order_date TARIH,
(DATEDIFF(DAY,first_order_date, last_order_date)) / (order_num_total_ever_offline + order_num_total_ever_online) ALISVERIS_SIKLIGI
FROM CUSTOMERS
WHERE first_order_date = (SELECT MIN(first_order_date) FROM CUSTOMERS)

/*
    MUSTERI_ID	                                TARIH	    ALISVERIS_SIKLIGI
    1d033d3a-b090-11e9-9757-000d3a38a36f	2013-01-14	59.13725490196078431372
*/