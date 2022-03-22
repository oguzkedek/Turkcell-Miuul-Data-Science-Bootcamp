## İŞ PROBLEMİ ARL RECOMMENDER SYSTEM 

Aşağıda 3 farklı kullanıcının sepet bilgileri verilmiştir. Bu sepet bilgilerine 
en uygun ürün önerisini birliktelik kuralı kullanarak yapınız. 
Ürün önerileri 1 tane yada 1'den fazla olabilir. Karar kurallarını 2010-2011 
Germany müşterileri üzerinden türetiniz.


Kullanıcı1’in sepetinde bulunan ürünün id'si: 21987
Kullanıcı2’in sepetinde bulunan ürünün id'si: 23235
Kullanıcı3’in sepetinde bulunan ürünün id'si: 22747


### Veri Seti Hikayesi 

OnlineRetailII isimli veriseti İngiltere merkezli bir perakende şirketinin
01/12/2009-09/12/2011 tarihleri arasındaki online satış işlemlerini içeriyor.
Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu müşterisinin
toptancı olduğu bilgisi mevcuttur.

- InvoiceNo = Fatura Numarası  ( Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini
ifade eder)
- StockCode = Ürünkodu( Her bir ürün için eşsiz)
- Description = Ürün ismi
- Quantity = Ürünadedi( Faturalardaki ürünlerden kaçar tane satıldığı)
- InvoiceDate = Fatura tarihi
- UnitPrice = Fatura fiyatı ( Sterlin )
- CustomerID = Eşsiz müşteri numarası
- Country = Ülke ismi

## İŞ PROBLEMİ HYBRID RECOMMENDER SYSTEM 

ID'si verilen kullanıcı için item-based ve user-based recommender yöntemlerini 
kullanarak 10 film önerisi yapınız.

Veriseti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır. 
İçerisinde filmler ile birlikte bu filmlere yapılan derecelendirme puanlarını
barındırmaktadır.27.278 filmde 2.000.0263 derecelendirme içermektedir .Bu veriseti
ise 17 Ekim 2016 tarihinde oluşturulmuştur. 138.493 kullanıcı ve 09 Ocak 1995 
ile 31 Mart 2015 tarihleri arasında verileri içermektedir. Kullanıcılar rastgele 
seçilmiştir.Seçilen tüm kullanıcıların en az 20filme oy verdiği bilgisi mevcuttur.

- movieId = Eşsiz film numarası.
- title = Film adı
- genres = Tür 
- userid = Eşsiz kullanıcı numarası. (UniqueID)
- movieId = Eşsiz film numarası. (UniqueID)
- rating = Kullanıcı tarafından filme verilen puan 
- timestamp = Değerlendirme tarihi
