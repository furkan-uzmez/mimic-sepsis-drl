from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

OUT = Path("iql_final_presentation.pptx")
FIG = Path("figures")

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

INK = RGBColor(17, 19, 22)
PAPER = RGBColor(244, 240, 229)
SIGNAL = RGBColor(226, 61, 40)
MINT = RGBColor(147, 217, 200)
STEEL = RGBColor(98, 113, 122)
WHITE = RGBColor(255, 255, 255)

W = prs.slide_width
H = prs.slide_height


def rgb_fill(shape, color, transparency=0):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.fill.transparency = transparency
    shape.line.color.rgb = INK
    shape.line.width = Pt(1)


def add_bg(slide, kicker):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = PAPER
    # Border frame
    frame = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.22), Inches(0.22), W - Inches(0.44), H - Inches(0.44))
    frame.fill.background()
    frame.line.color.rgb = RGBColor(185, 179, 165)
    frame.line.width = Pt(1)
    # Signal corner tag
    tag = slide.shapes.add_textbox(Inches(9.55), Inches(0.38), Inches(3.25), Inches(0.28))
    p = tag.text_frame.paragraphs[0]
    p.text = kicker.upper()
    p.alignment = PP_ALIGN.RIGHT
    r = p.runs[0]
    r.font.name = "Space Mono"
    r.font.size = Pt(8.5)
    r.font.bold = True
    r.font.color.rgb = SIGNAL
    return slide


def textbox(slide, text, x, y, w, h, size=20, color=INK, bold=False, font="Nunito", align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    r = p.runs[0]
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return box


def title(slide, text, x=0.75, y=1.05, w=5.6, h=1.6, size=34):
    box = textbox(slide, text.upper(), x, y, w, h, size=size, bold=True, font="Archivo")
    box.text_frame.paragraphs[0].line_spacing = 0.88
    return box


def eyebrow(slide, text, x=0.75, y=0.78, w=5.8):
    return textbox(slide, text.upper(), x, y, w, 0.25, size=9, color=SIGNAL, bold=True, font="Space Mono")


def body(slide, text, x, y, w, h, size=14, color=INK):
    return textbox(slide, text, x, y, w, h, size=size, color=color, font="Nunito")


def bullet_box(slide, items, x, y, w, h, fill=WHITE):
    rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    rgb_fill(rect, fill, 18 if fill == WHITE else 0)
    tf = rect.text_frame
    tf.clear()
    tf.margin_left = Inches(0.22)
    tf.margin_right = Inches(0.18)
    tf.margin_top = Inches(0.18)
    tf.margin_bottom = Inches(0.12)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"■ {item}"
        p.level = 0
        p.font.name = "Nunito"
        p.font.size = Pt(13)
        p.font.color.rgb = INK
        p.space_after = Pt(7)
        p.line_spacing = 1.08
    return rect


def metric(slide, value, label, x, y, w=1.55, h=0.72, fill=WHITE):
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    rgb_fill(box, fill, 12 if fill == WHITE else 0)
    tf = box.text_frame
    tf.clear()
    tf.margin_left = Inches(0.08)
    tf.margin_top = Inches(0.08)
    p = tf.paragraphs[0]
    p.text = str(value)
    p.font.name = "Archivo"
    p.font.bold = True
    p.font.size = Pt(22)
    p.font.color.rgb = SIGNAL
    p2 = tf.add_paragraph()
    p2.text = label.upper()
    p2.font.name = "Space Mono"
    p2.font.bold = True
    p2.font.size = Pt(7)
    p2.font.color.rgb = STEEL
    return box


def image_slide(kicker, heading, image_name, caption, note=None):
    s = add_bg(prs.slides.add_slide(prs.slide_layouts[6]), kicker)
    eyebrow(s, kicker)
    title(s, heading, size=31, w=4.6)
    if note:
        body(s, note, 0.8, 2.7, 4.25, 1.1, size=13, color=STEEL)
    img_path = FIG / image_name
    if img_path.exists():
        s.shapes.add_picture(str(img_path), Inches(5.55), Inches(1.05), width=Inches(6.9), height=Inches(5.2))
    textbox(s, caption.upper(), 5.55, 6.35, 6.9, 0.22, size=8, color=STEEL, bold=True, font="Space Mono")
    return s

# 1
s = add_bg(prs.slides.add_slide(prs.slide_layouts[6]), "Offline RL / Sepsis")
eyebrow(s, "MIMIC-IV + IQL")
title(s, "Sepsis Dozaj Politikaları", w=5.1, h=2.0, size=37)
body(s, "Destek sınırlı klinik koşullarda çevrimdışı pekiştirmeli öğrenme iş akışı.", 0.8, 3.45, 5.3, 0.8, 16, STEEL)
card = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.7), Inches(1.25), Inches(5.25), Inches(3.4))
rgb_fill(card, SIGNAL)
body(s, "Final aday", 7.05, 1.62, 4.7, 0.35, 18, WHITE)
body(s, "iql_sofa_shaped_conservative_safe", 7.05, 2.15, 4.7, 0.55, 18, WHITE)
metric(s, "2.848", "FQE", 7.05, 3.2, fill=WHITE)
metric(s, "8.203", "WIS", 8.82, 3.2, fill=WHITE)
metric(s, "29.41", "ESS", 10.58, 3.2, fill=WHITE)

# 2
s = add_bg(prs.slides.add_slide(prs.slide_layouts[6]), "Problem")
eyebrow(s, "Neden zor?")
title(s, "Klinik veri destek sınırı taşır", w=4.9)
bullet_box(s, ["Sepsis kararları gecikmeli mortalite sonucuna bağlanır.", "Daha ağır hastalar daha agresif tedavi aldığı için confounding oluşur.", "Çevrimdışı RL hastada keşif yapmaz; yalnız tarihsel yörüngeleri kullanır.", "Desteksiz sıvı-vazopresör kombinasyonları klinik olarak güvenilmezdir."], 6.0, 1.35, 5.9, 3.5)
textbox(s, "01", 9.3, 4.7, 2.7, 1.25, size=72, color=RGBColor(232, 164, 152), bold=True, font="Archivo")

# 3
s = add_bg(prs.slides.add_slide(prs.slide_layouts[6]), "Mimari")
eyebrow(s, "10 adımlı protokol")
title(s, "Model değil, denetlenebilir iş akışı", w=6.7, size=31)
steps = ["Kohort\nSepsis-3", "72 saat\n4 saatlik adım", "Patient split\ntrain-only fit", "62 boyutlu\nstate", "5x5 action\niki reward", "Replay\nbaseline", "IQL\neğitim", "OPE\nFQE/WIS/ESS", "Güvenlik\nteşhisleri", "Final sweep\nartifact"]
for i, st in enumerate(steps):
    x = 0.8 + (i % 5) * 2.42
    y = 3.05 + (i // 5) * 1.35
    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(2.05), Inches(1.02))
    rgb_fill(box, WHITE, 25)
    textbox(s, str(i+1), x+0.1, y+0.1, 0.35, 0.25, size=15, color=SIGNAL, bold=True, font="Archivo")
    body(s, st, x+0.48, y+0.16, 1.42, 0.6, size=10)

# 4
s = add_bg(prs.slides.add_slide(prs.slide_layouts[6]), "MDP")
eyebrow(s, "Formülasyon")
title(s, "Sonlu ufuklu karar süreci", w=5.2)
body(s, "Her geçiş hasta durumunu, 25 ayrık tedavi aksiyonunu ve terminal/ara ödülü taşır.", 0.8, 2.75, 5.0, 0.8, 15, STEEL)
code = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.05), Inches(1.35), Inches(5.6), Inches(1.3))
rgb_fill(code, INK)
body(s, "M = (S, A, P, R, gamma)\n(s_t, a_t, r_t, s_{t+1}, done_t)", 6.32, 1.62, 5.0, 0.75, 16, MINT)
metric(s, "62", "state feature", 6.05, 3.1)
metric(s, "25", "action bin", 7.92, 3.1)
metric(s, "0.99", "gamma", 9.8, 3.1)

# 5 action grid
s = add_bg(prs.slides.add_slide(prs.slide_layouts[6]), "Action Space")
eyebrow(s, "Sıvı x Vazopresör")
title(s, "25 ayrık tedavi kutusu", w=5.0)
bullet_box(s, ["Bin 0: tedavi yok.", "Sıfır dışı dozlar eğitim bölmesi çeyreklikleriyle Q1-Q4.", "Eşikler train üzerinde öğrenilir, validation/test üzerinde dondurulur."], 0.82, 3.2, 4.7, 2.0)
for r in range(5):
    for c in range(5):
        idx = r*5+c
        x, y = 6.35 + c*0.82, 1.34 + r*0.82
        cell = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.68), Inches(0.68))
        rgb_fill(cell, RGBColor(250, 220, 214) if idx in (0, 6, 12, 18, 24) else WHITE, 8)
        textbox(s, str(idx), x+0.16, y+0.2, 0.35, 0.2, size=13, color=INK, bold=True, font="Archivo", align=PP_ALIGN.CENTER)

# 6 rewards
s = add_bg(prs.slides.add_slide(prs.slide_layouts[6]), "Rewards")
eyebrow(s, "Ödül tasarımı")
title(s, "Seyrek sonuca SOFA şekillendirmesi eklenir", w=6.4, size=30)
for x, h, t in [(0.9, "Sparse", "90 gün sağkalım +15, 90 gün mortalite -15."), (4.65, "SOFA-shaped", "Terminal fayda korunur; ara adımlarda SOFA değişimi küçük sinyal üretir."), (8.4, "Kontrol", "Shaping mortalite hedefinin yerini almaz; final seçim ortak terminal değerlendirmeye bağlanır.")]:
    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(3.1), Inches(3.25), Inches(1.75))
    rgb_fill(box, WHITE, 20)
    textbox(s, h, x+0.2, 3.32, 2.8, 0.35, size=18, bold=True, font="Archivo")
    body(s, t, x+0.2, 3.9, 2.8, 0.8, 12)

# 7 IQL
s = add_bg(prs.slides.add_slide(prs.slide_layouts[6]), "IQL")
eyebrow(s, "Algoritma seçimi")
title(s, "IQL veri dışı aksiyon maksimumu almaz", w=5.7, size=31)
bullet_box(s, ["Value: expectile regression ile veri içindeki yüksek değerli aksiyonlara yaklaşır.", "Q: öğrenilen state-value hedefiyle güncellenir.", "Actor: avantaj ağırlıklı davranış klonlama ile politika çıkarır.", "Temperature arttıkça getiri potansiyeli ve destek dışına kayma riski birlikte artar."], 6.0, 1.45, 5.9, 3.5)
textbox(s, "IQL", 8.4, 4.9, 3.4, 1.0, size=72, color=RGBColor(232, 164, 152), bold=True, font="Archivo")

# 8 sweep
s = add_bg(prs.slides.add_slide(prs.slide_layouts[6]), "Sweep")
eyebrow(s, "Final tarama")
title(s, "18 aday, 6 finalist, 3 tohum", w=6.5)
items = [("2 ödül", "Sparse ve SOFA-shaped aynı terminal değerlendirme altında karşılaştırıldı."), ("3 LR rejimi", "Konservatif, referans ve aktör-konservatif optimizasyon."), ("3 IQL ayarı", "Güvenli, referans ve iyimser expectile-temperature çiftleri."), ("3 seed", "Finalistler 42, 123 ve 456 tohumlarıyla yeniden ölçüldü.")]
for i, (h, t) in enumerate(items):
    x = 0.9 + (i % 2) * 5.5
    y = 3.05 + (i // 2) * 1.35
    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(4.75), Inches(1.0))
    rgb_fill(box, WHITE, 22)
    textbox(s, h, x+0.18, y+0.14, 1.5, 0.3, size=15, bold=True, font="Archivo")
    body(s, t, x+1.72, y+0.14, 2.75, 0.55, 10.5)

# 9 stage 1
s = add_bg(prs.slides.add_slide(prs.slide_layouts[6]), "Stage 1")
eyebrow(s, "Finalist seçimi")
title(s, "Yüksek değer tek başına yeterli değil", w=5.4, size=31)
body(s, "Seçim FQE/WIS ile destek, uyum, düşük destek uyarısı ve çeşitlilik slotlarını birlikte okur.", 0.82, 2.95, 4.85, 0.9, 14, STEEL)
rows = [["Aday", "FQE", "WIS", "Destek"], ["SOFA conservative safe", "3.169", "8.616", "0.965"], ["SOFA conservative baseline", "2.280", "7.009", "0.973"], ["Sparse conservative safe", "1.954", "8.740", "0.971"], ["Sparse baseline safe", "2.345", "8.125", "0.963"]]
table = s.shapes.add_table(len(rows), 4, Inches(5.8), Inches(1.55), Inches(6.35), Inches(2.7)).table
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = table.cell(r, c)
        cell.text = val
        cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if r else INK
        for p in cell.text_frame.paragraphs:
            p.font.name = "Nunito"; p.font.size = Pt(10); p.font.bold = r == 0; p.font.color.rgb = WHITE if r == 0 else (SIGNAL if c == 3 and r else INK)

# images slides
image_slide("Evidence", "Değer ve destek birlikte raporlanır", "fqe_vs_support.png", "FQE vs support", "Yüksek değer tahminleri destek kütlesiyle birlikte yorumlanır.")

# 11 stage 2 table
s = add_bg(prs.slides.add_slide(prs.slide_layouts[6]), "Stage 2")
eyebrow(s, "Üç tohumlu sonuç")
title(s, "Final: SOFA + konservatif + güvenli", w=6.5, size=31)
rows = [["Sıra", "Konfigürasyon", "Skor", "FQE", "WIS", "ESS", "Uyum"], ["1", "SOFA konservatif güvenli", "5.858", "2.848", "8.203", "29.41", "0.414"], ["2", "SOFA konservatif referans", "5.288", "2.366", "8.286", "26.12", "0.404"], ["3", "Sparse konservatif güvenli", "5.262", "2.316", "8.169", "31.32", "0.411"], ["4", "Sparse konservatif referans", "5.033", "2.050", "8.536", "26.10", "0.401"]]
table = s.shapes.add_table(len(rows), 7, Inches(0.82), Inches(3.0), Inches(11.75), Inches(2.1)).table
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = table.cell(r, c); cell.text = val
        cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if r else INK
        for p in cell.text_frame.paragraphs:
            p.font.name = "Nunito"; p.font.size = Pt(8.5); p.font.bold = r == 0; p.font.color.rgb = WHITE if r == 0 else (SIGNAL if c in (2, 5) and r else INK)

image_slide("Baselines", "Davranış klonlama yüksek WIS üretir ama ESS çöker", "baseline_comparison.png", "Baseline comparison", "Clinician ESS=2585, BC ESS=1.6, IQL ESS=29.4.")
image_slide("Safety", "Politika klinisyeni kopyalamaz, ama destek içinde kalır", "action_heatmap.png", "Action heatmap", "Destek kütlesi 0.991; tam-kutu uyum 0.414; desteklenmeyen sapma 0.009.")
image_slide("Uncertainty", "CI geniş, ESS düşük", "bootstrap_ci.png", "Bootstrap WIS CI", "WIS 95% CI = 4.963--10.817. Sonuç klinik üstünlük iddiası değildir.")

# stack
s = add_bg(prs.slides.add_slide(prs.slide_layouts[6]), "Stack")
eyebrow(s, "Yeniden üretilebilirlik")
title(s, "Artifact odaklı deney yığını", w=5.6)
for i, (h, t) in enumerate([("Data", "Polars, PyArrow, Parquet ve hasta düzeyi manifestolar."), ("Training", "PyTorch, özelleştirilmiş d3rlpy, CUDA/MPS/auto cihaz seçimi."), ("Config", "Hydra, run naming, seed ve checkpoint sözleşmeleri."), ("Tracking", "MLflow, grafik paketi ve IEEE rapor artifact'leri.")]):
    x = 6.0 + (i % 2) * 2.95
    y = 1.55 + (i // 2) * 1.65
    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(2.6), Inches(1.22))
    rgb_fill(box, WHITE, 22)
    textbox(s, h, x+0.16, y+0.15, 2.2, 0.3, size=16, bold=True, font="Archivo")
    body(s, t, x+0.16, y+0.55, 2.18, 0.55, 9.5)

# ethics
s = add_bg(prs.slides.add_slide(prs.slide_layouts[6]), "Ethics")
eyebrow(s, "Sınırlar")
title(s, "Bu bir yatak başı tedavi önerisi değildir", w=5.8, size=31)
bullet_box(s, ["Çalışma retrospektif ve gözlemseldir.", "Durum vektörü hekim niyetini ve ölçülmeyen karıştırıcıları tam kapsamaz.", "FQE/WIS prospektif klinik kanıt değildir.", "MIMIC-IV kullanımı PhysioNet, CITI ve DUA sınırları içinde kalmalıdır."], 6.0, 1.35, 5.9, 3.5)

# future
s = add_bg(prs.slides.add_slide(prs.slide_layouts[6]), "Future Work")
eyebrow(s, "Sonraki araştırma")
title(s, "Daha büyük grid değil, belirsizlik-duyarlı politika güncellemesi", w=7.4, size=29)
for x, h, t in [(0.9, "Dinamik expectile", "Yoğun destekte agresif, seyrek bölgede davranışa yakın güncelleme."), (4.65, "Multi-step traces", "Seyrek terminal mortalite sinyalini yörünge boyunca daha hızlı taşıma."), (8.4, "World model", "Gerçekçi sınır durumlarını sentezleyerek değer fonksiyonuna güvenli marj öğretme.")]:
    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(3.45), Inches(3.25), Inches(1.55))
    rgb_fill(box, WHITE, 20)
    textbox(s, h, x+0.18, 3.62, 2.8, 0.3, size=16, bold=True, font="Archivo")
    body(s, t, x+0.18, 4.05, 2.8, 0.58, 10.5)

# takeaway
s = add_bg(prs.slides.add_slide(prs.slide_layouts[6]), "Takeaway")
eyebrow(s, "Son mesaj")
title(s, "Destek sınırı görünürse sonuç daha dürüst olur", w=5.9, size=34)
card = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.65), Inches(1.55), Inches(5.25), Inches(2.6))
rgb_fill(card, SIGNAL)
body(s, "Final okuma", 7.0, 1.9, 4.6, 0.35, 18, WHITE)
body(s, "IQL iş akışı klinik üstünlük değil; sızıntısız, destek-duyarlı, retrospektif politika dışı değerlendirme kanıtı üretir.", 7.0, 2.48, 4.5, 1.0, 16, WHITE)

prs.save(OUT)
print(OUT.resolve())
