## İŞ PROBLEMİ

Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli 
geliştirilmesi beklenmektedir.

Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev 
telefonu ve İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında 
bilgi içerir. Hangi müşterileri nhizmetlerinden ayrıldığını,kaldığını veya hizmete
kaydolduğunu gösterir.

- CustomerId = Müşteri Id’si  
- Gender = Cinsiyet  
- SeniorCitizen = Müşterinin yaşlı olu olmadığı(1, 0)  
- Partner = Müşterinin bir ortağı olup olmadığı(Evet, Hayır)  
- Dependent s = Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı(Evet, Hayır)  
- tenure = Müşterinin şirkette kaldığı ay sayısı  
- PhoneService = Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)  
- Multiple Lines = Müşterinin birden fazla hattı olup olmadığı(Evet, Hayır, Telefonhizmetiyok)  
- InternetService = Müşterinin internet servis sağlayıcısı(DSL, Fiber optik, Hayır)  
- OnlineSecurity = Müşterinin çevrimiçi güvenliğinin olup olmadığı(Evet, Hayır, İnternet hizmetiyok)  
- OnlineBackup = Müşterinin online yedeğinin olup olmadığı(Evet, Hayır, İnternet hizmetiyok)  
- DeviceProtection = Müşterinin cihaz korumasına sahip olup olmadığı(Evet, Hayır, İnternet hizmetiyok)  
- TechSupport = Müşterinin teknik destek alıp almadığı(Evet, Hayır, İnternet hizmetiyok)  
- StreamingTV = Müşterinin TV yayını olup olmadığı(Evet, Hayır, İnternet hizmeti yok)  
- StreamingMovies = Müşterinin film akışı olup olmadığı(Evet, Hayır, İnternet hizmetiyok)  
- Contract = Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)  
- PaperlessBilling = Müşterinin kağıtsız faturası olup olmadığı(Evet, Hayır)  
- PaymentMethod = Müşterinin ödeme yöntemi(Elektronikçek, Posta çeki, Banka havalesi(otomatik), Kredikartı(otomatik))  
- MonthlyCharges = Müşteriden aylık olarak tahsil edilen tutar  
- TotalCharges = Müşteriden tahsil edilen toplam tutar  
- Churn = Müşterinin kullanıp kullanmadığı(Evet veya Hayır)