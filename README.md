# ⌨️ Keystroke Dynamics Continuous Authentication with Bi-LSTM & Metric Learning

Bu proje, klavye kullanım alışkanlıklarını (Keystroke Dynamics) analiz ederek sürekli kimlik doğrulama (Continuous Authentication) ve Sıfır Güven Mimarisi (Zero-Trust Security) sağlayan derin öğrenme tabanlı bir davranışsal biyometri sistemidir.

## 🎯 Proje Amacı ve Çözülen Problem
Şifrelerin çalınabildiği veya paylaşılabildiği günümüzde, sisteme giren kişinin *gerçekten* yetkili kişi olup olmadığını sürekli denetlemek kritiktir. Bu proje, kullanıcının klavyedeki basılı tutma (Dwell) ve tuşlar arası uçuş (Flight) sürelerini analiz ederek arka planda görünmez bir güvenlik katmanı oluşturur.

## 🗄️ Veri Seti (Dataset)
Modelin eğitiminde endüstri standardı olan geniş çaplı Keystroke Dynamics veri seti kullanılmıştır. Bu veri seti, binlerce kullanıcının serbest metin yazma (free-text) sırasındaki tuş basma ve bırakma (milisaniye) loglarını içerir.
*   **Veri Seti Kaynağı:** [Keystroke Dynamics Benchmark Dataset (IEEE DataPort / V. Monaco)](https://ieee-dataport.org/open-access/keystroke-dynamics-benchmark-dataset)
*   **Veri Boyutu:** +168.000 ham `.txt` kayıt dosyası.
*   **İçerik:** `Key` (Basılan Tuş), `Press_Time` (Basılma Anı), `Release_Time` (Bırakılma Anı).

## 🛡️ Veri Gizliliği (Privacy by Design) ve 16 Bölge (Zone) Yaklaşımı
Bu projenin en kritik mühendislik kararlarından biri, veri toplama aşamasında **gizlilik (privacy)** ihlallerini önlemektir. 30 gün boyunca basılan tuşların doğrudan kaydedilmesi (Keylogging); şifrelerin ve özel yazışmaların açığa çıkmasına neden olur.

**Çözüm:** Klavye 16 ergonomik bölgeye (Zone) ayrılmıştır. Sistem kullanıcının spesifik olarak hangi tuşa bastığını değil; *"1. Bölgeden 3. Bölgeye geçerken ne kadar süre harcadığını"* kaydeder. Böylece kullanıcıların mahremiyeti %100 korunurken, biyometrik ritim haritası başarıyla çıkarılmıştır.

## ⚙️ Veri Ön İşleme ve Mühendislik (Data Preprocessing)
+168.000 dosyalık devasa veriyi standart bilgisayar belleğine (RAM) sığdırmak imkansız olduğu için, projede ileri düzey veri mühendisliği teknikleri uygulanmıştır:

1.  **Özellik Çıkarımı (Feature Engineering):** 
    *   **Dwell Time (Basılı Tutma):** `Release_Time - Press_Time` formülüyle hesaplandı.
    *   **Flight Time (Uçuş Süresi):** Bir sonraki tuşun basılma anı ile mevcut tuşun bırakılma anı arasındaki fark olarak ölçüldü.
2.  **Label Encoding:** 16 farklı klavye bölgesi arasındaki geçişler (Örn: 1_3, 4_12) hesaplandı ve `LabelEncoder` ile 16x16 = 256 kelimelik bir "geçiş sözlüğüne" (Vocabulary) dönüştürüldü.
3.  **Out-of-Core Processing (Bellek Dışı İşleme):** Veriler tek seferde RAM'e yüklenmedi. 1000 dosyadan oluşan paketlere (Chunk) bölündü. `DataLoader` sadece o an eğitilen paketi belleğe aldı.
4.  **Local Standard Scaling:** Modelin gradient patlaması yaşamaması için, her bir 1000'lik chunk kendi içinde `StandardScaler` ile ölçeklendirildi (Mean=0, Std=1).
5.  **Dinamik Eşleştirme:** Siamese ağı için veriler %50 ihtimalle aynı kişinin (Label 1), %50 ihtimalle farklı kişilerin (Label 0) 20 tuşluk ardışık dizileri olarak eşleştirildi.

## 🧬 Projenin Mimari Evrimi (V2 - V6)
Bu proje, istatistiksel darboğazların aşıldığı iteratif bir Ar-Ge sürecidir:

*   **V2 & V3 (Baseline Sınıflandırma):** İlk denemelerde basit YSA kullanılmış, ancak bellek taşması ve Gradient kaybolması (Dying ReLU) sorunları yaşanmıştır.
*   **V4 (Siamese Network + BCE Loss):** İkili sınıflandırma (Binary Classification) mantığına geçildi. Başlangıçta Loss değeri `0.69`'da tıkandı. Dinamik Batch ölçeklendirme ile model körlükten kurtarıldı. **EER (Equal Error Rate): %26.67** olarak ölçüldü.
*   **V5 (Metric Learning'e Geçiş):** Yüz tanıma sistemlerinin standardı olan **Contrastive Loss (Zıtlık Kaybı)** ve L2 Normalizasyonu kullanıldı. Model ritimleri 64 boyutlu uzayda koordinatlara çevirdi. EER %20.00'a düştü, ancak test setinin 50 kişi olması istatistiksel dalgalanmalara (variance) yol açtı.
*   **V6 (Bi-LSTM + Devasa Test Seti - Final):** Ağ yapısı *Çift Yönlü LSTM (Bi-LSTM)* olarak güncellendi. Model artık sadece basılan tuşu değil, *bir sonraki basılacak tuşun* anatomik etkisini de öğreniyor. İstatistiksel güvenilirlik için test seti 2000 kişiye çıkarıldı.

## 🚀 Final Performansı (V6 Sonuçları)
Model, eğitimde hiç görmediği 2000 kişilik devasa test setinde test edilmiştir.

*   **Optimal Eşik Doğruluğu:** %88.08
*   **ROC-AUC Skoru:** 0.9504 (State-of-the-art seviyesinde uzay ayrımı)
*   **EER (Equal Error Rate):** %12.21
*   **Gerçek Dünya Simülasyonu (Sliding Window / Çoğunluk Oyu):** Sistem tek bir tuş geçişinde %12 yanılma payına sahipken; saldırganın peş peşe 5, 10 veya 15 tuş girdiği (Kayan Pencere) senaryolarında doğruluk oranı **%100.00**'a (Kesin Tespit) ulaşmıştır.

## 🛠️ Kullanılan Teknolojiler
*   **Derin Öğrenme:** PyTorch (Bi-LSTM, Siamese Networks, Metric Learning / Contrastive Loss)
*   **Veri İşleme:** Pandas, NumPy, Scikit-learn (Label Encoding, Standard Scaling)
*   **Altyapı:** Google Colab (CUDA), Out-of-Core Pipeline Yönetimi

## 🌍 Gerçek Hayat Kullanım Alanları
1.  **Kurumsal Sürekli Doğrulama (Continuous Authentication):** Çalışan şifresiyle giriş yapsa bile, bilgisayarın başına başka biri geçtiğinde klavye ritmi değişeceğinden sistemin kendini anında kilitlemesi.
2.  **Uzaktan Sınav/Çalışma Güvenliği:** Ekranın başındaki kişinin başkasından destek alıp almadığının veya yer değiştirip değiştirmediğinin tespiti.
3.  **Zero-Trust Security Mimarisi:** Sadece parolaya değil, "parolayı yazan elin anatomisine" güvenen modern siber güvenlik sistemleri.