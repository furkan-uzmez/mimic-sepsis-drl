# Proje Önerisi Geri Bildirimi

> **Ders:** BLM446 Pekiştirmeli Öğrenme  
> **Aşama:** Konu Önerisi (Hafta 6 teslimi)  
> **Geri bildirim tarihi:** 14 Mayıs 2026

---

## Önerinizin Özeti

MIMIC-IV sepsis kohortu üzerinde geçmiş klinik kayıtlardan öğrenen bir çevrimdışı pekiştirmeli öğrenme (offline RL) politikası geliştirilmesi hedeflenmektedir. Her dört saatlik klinik pencerede uygun vazopressör ve damar içi sıvı kararı önerilecektir. 62-boyutlu sürekli klinik durum vektörü, 25 ayrık eylem (5×5 ızgara) ve CQL (Conservative Q-Learning) algoritması seçilmiş; BCQ ve IQL alternatif olarak incelenmiştir.

## Güçlü Yönler

- **MDP formülasyonu** — 62-boyutlu durum vektörünün bileşenleri kategorize edilmiş (vital bulgular, laboratuvar değerleri, ventilatör parametreleri, demografik veriler, türetilmiş özellikler); eylem uzayı 5×5 ızgara yapısıyla gerekçelendirilmiş; SOFA tabanlı ara ödül ve terminal ödüller tanımlanmış; BLM446 lisans beklentisini belirgin biçimde aşan bir ayrıntı düzeyi.
- **Algoritma seçim gerekçesi** — CQL'in çevrimdışı RL'deki korumacı değer tahmini özelliği ve ayrık eylem uzayıyla uyumu akademik olarak tutarlı biçimde açıklanmış; IQL ve BCQ ile karşılaştırmalı üstünlük tartışılmış.
- **Ayrık eylem seçim argümanı** — Çevrimdışı dağılım dışı eylem (out-of-distribution action) riski, seyrek veri ve hesaplama maliyeti gibi somut gerekçeler sunulmuş.
- **Değerlendirme protokolü** — Çevrimdışı politika değerlendirme (off-policy evaluation, OPE) yöntemleri olan WIS, ESS ve FQE planlanmış; bu düzey metodolojik farkındalık öne çıkıyor.
- **Baseline planı** — Klinisyen tekrarı (clinician replay), tedavisiz kontrol ve davranış klonlama (behavior cloning) baseline'ları yazılmış; OPE protokolüyle birlikte karşılaştırma zeminini somutlaştırıyor.

## Gelişim Alanları

### Final Teslim Öncesi Tamamlanması Önerilen Noktalar

1. **İndirim faktörünü gerekçelendirin** — $\gamma$ değeri öneri metninde belirtilmemiştir. Sepsis bölümlerinin uzunluğunu (~4 saat × N adım) dikkate alarak $\gamma = 0.99$ gibi bir değeri gerekçesiyle seçin.
2. **Gözlemlenebilirlik tartışması** — Klinik kayıtlarda eksik veri ve gizli değişkenler (örn. ölçülemeyen enflamasyon belirteçleri) MDP'yi kısmi gözlemlenebilir (POMDP) hale getirebilir; bu riski kısaca tartışın.
3. **Veri erişim sürecini belirtin** — MIMIC-IV PhysioNet üzerinden CITI eğitimi ve Veri Kullanım Sözleşmesi (DUA) gerektirmektedir; final raporunda etik bölümü eklenmelidir.
4. **Referans listesi** — Final için en az 10 IEEE formatında kaynak; Kumar 2020 (CQL), Fujimoto 2021 (BCQ), Kostrikov 2021 (IQL), Komorowski 2018 (AI Clinician) ve Levine 2020 (Offline RL tutorial) öncelikli olmalıdır.

### İsteğe Bağlı Geliştirmeler (Final Raporu Güçlendirir)

- **Baseline karşılaştırması:** Klinisyen ve davranış klonlama baseline'ları zaten planlanmış; bu bölüm güçlü.
- **Sayısal eşikli hipotez:** *"CQL, klinisyen baseline'a göre WIS normalize skorunda en az 0.1 mutlak iyileşme sağlar"* formatında somutlaştırılabilir.
- **Plan B / risk yönetimi:** *"CQL $\alpha_{\text{CQL}} = 5$ ile 50k adım sonra performans iyileşmezse IQL'ye geç"* gibi bir yedek senaryo projeyi daha sağlam kılar.
- **Referans listesi:** ≥ 5 IEEE formatında kaynak eklenmelidir.

## Durum

Önerinizin yöntem seçimi, MDP formülasyonu ve karşılaştırma planı projeyi yürütmek için yeterli temeli sunmaktadır. Konunuz **onaylanmıştır**; uygulamaya geçebilirsiniz.

## Sonraki Adımlar İçin Öneriler

1. $\gamma$ değerini gerekçeli olarak belirleyin; eksik veri ve POMDP riskini kısa bir paragrafla tartışın.
2. MIMIC-IV veri erişim sürecini (PhysioNet, CITI, DUA) etik bölümünde belgeleyin.
3. Referans listesini oluşturun ve isteğe bağlı geliştirmeleri (sayısal hipotez, Plan B) değerlendirin.

---

İyi çalışmalar dilerim.  
**Doç. Dr. Alpaslan Burak İNNER**  
Kocaeli Üniversitesi Bilgisayar Mühendisliği  
BLM446 Pekiştirmeli Öğrenme — 2025/2026 Bahar
