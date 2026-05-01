# Tema nr. 2: Document Scanning, Rectification and Quality Assessment

Echipa nr. 7:
- Danalache Emanuel
- Danalache Sebastian

## Scopul proiectului

Aplicatia implementeaza un scanner foto pentru documente care:
- detecteaza regiunea documentului in imagini reale cu fundal variat;
- rectifica perspectiva si produce o vedere top-down;
- imbunatateste lizibilitatea pentru printare sau OCR prin contrast local, de-shadowing si binarizare;
- calculeaza un scor de calitate pentru fiecare scanare;
- respinge automat scanarile de calitate slaba pe baza unui quality gate;
- ruleaza batch pe foldere si genereaza experimente simple pe conditii de fundal si iluminare.

## Status pe checkpoints

### Week 3
- dataset local in folderul `dataset/`;
- detectare de baza prin muchii, contururi si thresholding;
- rectificare de perspectiva functionala pentru imagini aproape frontale.

### Week 6
- detectare robusta pe fundal variat folosind mai multi detectori: `contour`, `threshold`, `hough`, `corners`;
- pipeline end-to-end: detectie -> rectificare -> binarizare -> metrici;
- metrici de baza pentru calitate: `sharpness`, `skew angle`, `contrast`, `brightness`, `detection confidence`, `overall score`.

### Week 9
- binarizare imbunatatita prin mai multe metode comparabile:
  - `adaptive_gaussian`
  - `otsu_deshadow`
  - `sauvola_deshadow`
  - `hybrid_deshadow`
- de-shadowing integrat in pipeline prin corectie de iluminare bazata pe blur/division si morph close/subtract;
- selectie automata a celei mai bune binarizari pe baza unui `binary score`;
- procesare batch pe foldere;
- rapoarte initiale pe conditii de fundal si iluminare;
- fisiere CSV salvate separat de output-ul imaginilor;
- pentru fiecare imagine se salveaza implicit un singur fisier compus `*_pipeline.jpg` care contine toate etapele utile intr-o singura imagine.

### Week 12
- script robust de scanare cu `quality score` si status explicit `accepted/rejected`;
- respingere automata a inputurilor slabe pe baza unor praguri pentru `overall score`, `detection confidence`, `sharpness`, `skew` si `binarization score`;
- benchmark comparativ pentru detectorii de document (`contour`, `threshold`, `hough`, `corners`);
- benchmark-ul detectorilor foloseste implicit o binarizare fixa `otsu_deshadow` pentru comparatie fair si rulare mai rapida;
- rapoarte CSV comparative atat pentru binarizare, cat si pentru detectori.

## Structura proiectului

- `document_scanner.py` - procesare rapida pentru o imagine sau un folder.
- `run_pipeline.py` - ruleaza lotul complet si genereaza toate rapoartele.
- `generate_plots.py` - genereaza graficele de analiza.
- `compare_methods.py` - compara metodele de binarizare si salveaza rapoarte CSV.
- `generate_comparison_grid.py` - creeaza un grid vizual din imaginile tip pipeline.
- `dataset/` - setul de imagini de intrare.
- `output/` - exemplu de output pentru imagini si grafice.

## Pipeline implementat

### 1. Preprocesare
- conversie la grayscale;
- reducere de zgomot cu `bilateralFilter`;
- crestere de contrast local cu `CLAHE`.

### 2. Detectia documentului
Se ruleaza mai multi detectori, iar cel mai bun patrulater valid este ales automat:
- `detect_contour()` - Canny + contururi;
- `detect_threshold_quad()` - thresholding + morfologie;
- `detect_hough()` - linii dominante si intersectii;
- `detect_corners_harris()` - colturi puternice.

Fiecare candidat este validat geometric:
- convexitate;
- arie minima rezonabila;
- laturi suficient de mari;
- unghiuri plauzibile pentru un document.

### 3. Rectificare
- ordonarea colturilor in format `TL, TR, BR, BL`;
- calculul transformarii de perspectiva cu `cv2.getPerspectiveTransform`;
- warp in vedere top-down cu `cv2.warpPerspective`.

### 4. Imbunatatire pentru scan
- de-shadowing prin doua familii de metode;
- binarizare multipla:
  - adaptiva `Gaussian`;
  - `Otsu` dupa de-shadowing;
  - `Sauvola` dupa de-shadowing;
  - varianta `hybrid` (local + global);
- selectie automata a metodei castigatoare pentru fiecare imagine;
- curatare morfologica `open + close`.

### 5. Evaluare calitate
- `sharpness`: varianta Laplacianului;
- `skew_angle`: estimat din liniile Hough;
- `contrast`: abaterea standard RMS;
- `brightness`: media intensitatii;
- `shadow_level`: variatia iluminarii;
- `binarization_score`: scor euristic pentru lizibilitatea binarizarii;
- `detection_confidence`: incredere euristica a detectiei;
- `overall_score`: scor agregat pe 100.

### 5b. Quality gate si rejectie
- un scan este marcat `accepted` sau `rejected`;
- pragul default este `overall_score >= 60`, dar mai exista si verificari suplimentare pentru:
  - `detection_confidence`;
  - blur puternic;
  - skew mare;
  - binarizare slaba;
- pentru fiecare imagine se salveaza in JSON/CSV:
  - `quality_status`;
  - `rejected_input`;
  - `rejection_reason`;
  - `quality_flags`.

### 6. Vizualizare si rapoarte
- pentru fiecare imagine se genereaza implicit un singur fisier:
  - `*_pipeline.jpg` = original + detectie + preprocess + rectified + de-shadow + toate metodele de binarizare + rezultatul final intr-o singura imagine;
  - header-ul afiseaza si statusul `ACCEPTED` sau `REJECTED`;
- optional, se pot salva separat si etapele `rectified` si `binary`;
- `summary.csv` - rezultate per imagine;
- `binarization_comparison.csv` - scorurile tuturor metodelor pe fiecare imagine;
- `binarization_summary.csv` - medie si numar de "wins" per metoda;
- `experiment_success_rates.csv` - rate de succes vs fundal si iluminare;
- `detector_comparison.csv` - rezultate per imagine pentru fiecare detector fortat;
- `detector_summary.csv` - detectie, usable rate, scor mediu si wins pentru fiecare detector;
- `comparison_grid.png` - optional, grid vizual al pipeline-urilor;
- `plot_1` ... `plot_7` - optional, grafice pentru detectie, calitate si comparatii.

## Instalare

```bash
pip install -r requirements.txt
```

Optional, pentru imagini `HEIC/HEIF`:

```bash
pip install pillow-heif pillow
```

## Cum rulezi programul

### Varianta completa
Comanda recomandata:

```bash
python run_pipeline.py --input dataset --output output
```

Prin default:
- imaginile de pipeline si metricile JSON se salveaza in `output/`;
- fisierele CSV se salveaza separat in `output_csv/`;
- pentru fiecare document se salveaza un singur `*_pipeline.jpg`;
- scanurile slabe sunt marcate automat `rejected`;
- plot-urile si grid-urile extra nu se genereaza automat.

### Daca vrei alt folder pentru CSV-uri

```bash
python run_pipeline.py --input dataset --output output --csv-output reports
```

### Varianta simpla

```bash
python document_scanner.py dataset -o output
```

Si aici, prin default:
- `summary.csv` merge in `output_csv/`;
- pentru fiecare imagine se salveaza un singur `*_pipeline.jpg`.

## Optiuni utile

Procesare recursiva:

```bash
python document_scanner.py dataset -o output --recursive
```

Mod debug:

```bash
python document_scanner.py dataset -o output --debug
```

Fixarea unei metode de binarizare:

```bash
python document_scanner.py dataset -o output --binarization otsu_deshadow
```

Fixarea unui detector de document:

```bash
python document_scanner.py dataset -o output --detection hough
```

Schimbarea pragului de rejectie:

```bash
python run_pipeline.py --input dataset --output output --quality-threshold 65
```

Dezactivarea rejectiei automate:

```bash
python run_pipeline.py --input dataset --output output --no-reject-low-quality
```

Salvarea tuturor variantelor de binarizare:

```bash
python run_pipeline.py --input dataset --output output --save-all-binaries
```

Salvarea si a etapelor separate (`rectified` si `binary`):

```bash
python run_pipeline.py --input dataset --output output --save-stages-separately
```

Generarea optionala a plot-urilor:

```bash
python run_pipeline.py --input dataset --output output --generate-plots
```

Generarea optionala a grid-ului global:

```bash
python run_pipeline.py --input dataset --output output --generate-grid
```

## Scripturi separate

Generarea graficelor:

```bash
python generate_plots.py --output output --csv-input output_csv
```

Comparatia metodelor de binarizare:

```bash
python compare_methods.py --output output --csv-output output_csv
```

Comparatia completa, fara benchmark-ul detectorilor:

```bash
python compare_methods.py --output output --csv-output output_csv --skip-detector-benchmark
```

La benchmark-ul detectorilor:
- se afiseaza progres pentru fiecare imagine si fiecare detector;
- se afiseaza un `ETA` aproximativ;
- se foloseste implicit `otsu_deshadow` ca binarizare fixa pentru benchmark;
- `detector_comparison.csv` si `detector_summary.csv` se actualizeaza incremental in timpul rularii.

## Output generat

### In folderul de imagini, de exemplu `output/`
- `*_pipeline.jpg` - pipeline complet intr-o singura imagine, cu toate etapele importante si status `ACCEPTED/REJECTED`;
- `*_metrics.json` - metrici + etichete de scena + scoruri pe metode + quality gate;

Optional, daca folosesti `--save-stages-separately`:
- `*_rectified.jpg`;
- `*_binary.jpg`.

Optional, daca folosesti `--save-all-binaries`:
- `*_binary_<metoda>.jpg`.

Optional, daca folosesti `--generate-grid`:
- `comparison_grid.png`.

Optional, daca folosesti `--generate-plots`:
- `plot_1_detection_rate.png`;
- `plot_2_sharpness.png`;
- `plot_3_skew_angle.png`;
- `plot_4_overall_scores.png`;
- `plot_5_quality_breakdown.png`;
- `plot_6_success_by_condition.png`;
- `plot_7_binarization_methods.png`.

### In folderul de CSV-uri, de exemplu `output_csv/`
- `summary.csv`;
- `binarization_comparison.csv`;
- `binarization_summary.csv`;
- `experiment_success_rates.csv`.
- `detector_comparison.csv`;
- `detector_summary.csv`.

## Observatii

- Daca documentul nu poate fi detectat robust, sistemul foloseste imaginea intreaga ca fallback.
- Clasificarea pe `background_complexity` si `lighting_condition` este euristica, utila pentru experimente initiale, nu etichetare manuala.
- Pentru imagini deja scanate sau perfect frontale, fallback-ul poate produce in continuare un rezultat bun.
- Chiar si pentru imagini rejectate, se salveaza `*_pipeline.jpg` pentru inspectie vizuala, dar statusul apare explicit in preview si in rapoartele CSV/JSON.
