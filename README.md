# TPMS & Metamaterial Stiffness NN Studio (SEM-3)

Interaktywna platforma inżynierska oparta na **Streamlit**, **PyTorch** oraz silniku geometrycznym **OpenCASCADE (OCP)**, służąca do modelowania i mikromechanicznej analizy relacji pomiędzy parametrami topologicznymi struktur TPMS, kratowych i auksetycznych a macierzą sztywności 6×6 w notacji Voigta, wraz z **generatorem i eksporterem modeli 3D w formatach inżynierskich CAD (STEP, IGES) i druku 3D (STL, OBJ, PLY)**.

Dostęp online: **[https://gyro.ignatech.pl](https://gyro.ignatech.pl)**

---

## 🚀 Architektura i Zaimplementowane Struktury

### 1. Siedem Kanonicznych Geometrii Komórek:
- **🧊 Schoen Gyroid ($Ia\bar{3}d$, nr 230)**:
  $$f_G(x, y, z) = \sin(k_x x)\cos(k_y y) + \sin(k_y y)\cos(k_z z) + \sin(k_z z)\cos(k_x x) - t = 0$$
  Trójwymiarowa struktura chiralna o zerowej średniej krzywiźnie ($H=0$), tworząca dwa splecione bezkontaktowe labirynty. Doskonała do wymienników ciepła i implantów osteokonduktywnych.
- **💎 Schwarz Diamond ($Pn\bar{3}m$, nr 224)**:
  $$f_D(x, y, z) = \sin(k_x x)\sin(k_y y)\sin(k_z z) + \sum \sin(k_x x)\cos(k_y y)\cos(k_z z) - t = 0$$
  Czworościenna koordynacja węzłów ($sp^3$), wysoka odporność na ściskanie osiowe i podwyższony moduł ścinania ($G/E \approx 0.41$).
- **📦 Schwarz Primitive ($Im\bar{3}m$, nr 229)**:
  $$f_P(x, y, z) = \cos(k_x x) + \cos(k_y y) + \cos(k_z z) - t = 0$$
  Dopełnienie "Wielkiej Trójki" TPMS. Regularna sieć kubiczna z trzema prostopadłymi kanałami o najniższych oporach przepływu medium.
- **🎀 Re-entrant Auxetic (Auksetyk 3D, $\nu < 0$)**:
  $$f_{aux}(x, y, z) = \cos(k_x x)\cos(k_y y)\cos(k_z z) - c\cdot(\cos(2k_x x) + \cos(2k_y y) + \cos(2k_z z)) - t = 0$$
  Metamateriał o **ujemnym współczynniku Poissona ($\nu < 0$)**. Wklęsłe krawędzie komórki rozszerzają się poprzecznie przy rozciąganiu wzdłużnym, dając bezkonkurencyjne właściwości pochłaniania energii uderzenia i odporności na pękanie.
- **🕸️ Neovius (Schoen C(P), $Im\bar{3}m$, nr 229)**:
  $$f_N(x, y, z) = \frac{1}{3}\left[3(\cos(k_x x) + \cos(k_y y) + \cos(k_z z)) + 4\cos(k_x x)\cos(k_y y)\cos(k_z z)\right] - t = 0$$
  Powierzchnia minimalna z 12 diagonalnymi kanałami w komórce. Wykazuje podwyższoną sztywność na ścinanie ($\gamma_{shear} \approx 0.44$) oraz bardzo dużą powierzchnię właściwą.
- **🔀 Schoen I-WP (I-Wrapped Package, $Im\bar{3}m$, nr 229)**:
  $$f_{IWP}(x, y, z) = 2(\cos(k_x x)\cos(k_y y) + \cos(k_y y)\cos(k_z z) + \cos(k_z z)\cos(k_x x)) - (\cos(2k_x x) + \cos(2k_y y) + \cos(2k_z z)) - t = 0$$
  Podwójny asymetryczny labirynt o zróżnicowanych objętościach komór i wysokiej quasi-izotropii mechanicznej ($\alpha \approx 1.12$).
- **🏗️ Octet-Truss (Schoen F-RD / Krata Oktetowa, $Fm\bar{3}m$, nr 225)**:
  $$f_{octet}(x, y, z) = 4\cos(k_x x)\cos(k_y y)\cos(k_z z) - (\cos(2k_x x)\cos(2k_y y) + \cos(2k_y y)\cos(2k_z z) + \cos(2k_z z)\cos(2k_x x)) - t = 0$$
  Płynna reprezentacja kraty oktetowej FCC (Deshpande-Fleck). Posiada mechanizm odkształceń zdominowany przez rozciąganie prętów (**stretching-dominated**, $m \approx 1.55$ w Network, $m \approx 1.10$ w Sheet), oferując najwyższy stosunek modułu Younga do gęstości.

---

### 2. Dwa Tryby Morfologii Komórki:
- **🧱 Tryb Network (Bryłowy / Szkieletowy)**:
  - Faza stała wypełnia jedną z komór labiryntu ($f(x, y, z) \le t$).
  - Mechanizm odkształcenia zdominowany przez zginanie ścianek (*bending-dominated*, $m \approx 1.85$).
- **📜 Tryb Sheet (Ściankowy / Powłokowy)**:
  - Materiał tworzy cienką powłokę o kontrolowanej grubości $2 \times t_w$ wokół powierzchni zerowej ($|f(x, y, z)| \le t_w$).
  - Mechanizm odkształcenia zdominowany przez rozciąganie ścianek w płaszczyźnie (*stretching-dominated*, $m \approx 1.20$).
  - Osiąga sztywność właściwą ($E/\rho$) zbliżoną do teoretycznej granicy Hashina-Shtrikmana (do $2\times$ wyższa sztywność niż Network przy tej samej masie).

---

### 3. Eksport Modeli 3D (CAD & Additive Manufacturing):
- **STEP (`.step` / `.stp`)**: Międzynarodowy standard ISO 10303 dla profesjonalnych systemów CAD (SolidWorks, Autodesk Inventor, Fusion 360, CATIA, Siemens NX, FreeCAD).
- **IGES (`.iges` / `.igs`)**: Klasyczny standard ANSI Y14.26M geometrii powierzchniowej CAD/CAM.
- **STL (`.stl`)**: Standard triangulacji (binarny) dla slicerów druku 3D (PrusaSlicer, Bambu Studio, Cura, OrcaSlicer).
- **OBJ (`.obj`) / PLY (`.ply`)**: Otwarte formaty siatek poligonowych dla grafiki komputerowej (Blender, MeshLab).
- Konfiguracja fizycznych wymiarów w milimetrach ($L_x \times L_y \times L_z$), rozdzielczości siatki oraz grubości ścianki $t_w$.

---

### 4. Zakładki i Narzędzia Interfejsu:
1. **🧊/💎/📦/🎀/🕸️/🔀/🏗️ 1. Morfologia & Modele 3D (7 Architektur)**:
   - Interaktywny podgląd Isosurface 3D w Plotly z regulacją przezroczystości, domknięć ścian (caps) i palet barwnych.
   - Tryb porównania wszystkich 7 struktur obok siebie (galeria z tabelą cech krystalograficznych i mechanicznych).
   - Szybki pobieracz plików 3D (STEP, IGES, STL, OBJ).
2. **📊 2. Macierze Sztywności 6×6**:
   - Zestawienie heatmap macierzy sztywności: Ground Truth ($C_{true}$), Predykcja Sieci Neuronowej ($C_{pred}$) oraz błąd bezwzględny $|C_{true} - C_{pred}|$.
   - Prezentacja ujemnych składowych sprzężenia poprzecznego $C_{12}, C_{13}, C_{23} < 0$ dla Auksetyka, podwyższonego ścinania Neoviusa oraz zachowania rozciąganego Octet-Truss.
3. **🌐 3. Kierunkowy Moduł Younga 3D**:
   - Trójwymiarowy wykres polarny tensora podatności czwartego rzędu $1/E(\mathbf{u}) = S_{ijkl} u_i u_j u_k u_l$.
   - Obliczenie wskaźnika anizotropii $E_{max} / E_{min}$.
4. **📈 4. Diagnostyka Uczenia Sieci**:
   - Wykresy historii straty (MSE) oraz współczynnika determinacji $R^2$.
5. **💾 5. Zbiór Danych & Generator CAD 3D**:
   - Eksport zbiorów danych do formatów `.csv` oraz `.npz`.
   - Pełnowymiarowy generator i konwerter modeli CAD w wybranym formacie (STEP/IGES/STL/OBJ/PLY) i wymiarach w mm.

---

## 🌐 Dostęp sieciowy i Reverse Proxy (Cosmos Cloud)
- **Domena publiczna**: [https://gyro.ignatech.pl](https://gyro.ignatech.pl)
- **Port w kontenerze / host**: `8501` (Streamlit)
- **Sieć Docker**: `cloudflare_net` oraz sieć domyślna
- **Reverse Proxy**: Cosmos Server (`azukaar/cosmos-server`) w trybie `SERVAPP`, przekierowujący ruch z domeny `gyro.ignatech.pl` na kontener `gyro-metamaterial-studio:8501`.

---

## 🐳 Uruchamianie za pomocą Dockera

W katalogu głównym projektu:

```bash
# Start w tle
docker compose up -d

# Podgląd logów
docker logs -f gyro-metamaterial-studio

# Zatrzymanie
docker compose down
```

---

## 💻 Uruchamianie lokalne (Python)

Wymagany Python 3.10+:

```bash
# Utworzenie i aktywacja środowiska wirtualnego
python3 -m venv .venv
source .venv/bin/activate

# Instalacja zależności
pip install -r requirements.txt

# Start aplikacji
streamlit run app.py
```
Aplikacja będzie dostępna pod adresem: `http://localhost:8501`.
