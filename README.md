# Súrlódási adatok elemzésére készített gépi tanulási pipeline

ENGLISH BELOW!

Ez a projekt egy olyan machine learning folyamat, amely tribométeres mérési adatok elemzésére, prediktív modellezésére és a kenési állapotok, súrlódási együtthatók optimalizálására szolgál. A kód kiegészítő, jövőbeli méréseket javasol a mérési paramétertér minél jobb lefedésére.

## Főbb funkciók
* **Adatfeldolgozás:** Zajszűrés (rolling mean), downsampling, és anomáliák (outlierek) automatikus eltávolítása.
* **Feature Engineering:** Fizikai paraméterek (pl. Hertz-féle feszültség) és interakciós jellemzők generálása, VIF (Variance Inflation Factor) alapú multikollinearitás szűrés.
* **Modell tanítás és optimalizálás:** 7 különböző algoritmus (XGBoost, LightGBM, CatBoost, Random Forest, MLP, KNN, Ridge) hiperparaméter-hangolása Optuna (Bayesian Optimization) segítségével.
* **Ensemble:** A legjobb 3 modell automatikus kiválasztása és súlyozott kombinálása.
* **Magyarázhatóság (XAI):** SHAP (SHapley Additive exPlanations) alapú ábrák a funkciók fontosságának és hatásának megértéséhez.
* **Optimum keresés és DoE:** Kompromisszumos (Pareto) optimumok meghatározása, valamint új mérési pontok javaslata (Active Learning / Sequential Design of Experiments).
* **Automatikus riportálás:** Átfogó HTML és Excel riport generálása a futás legvégén a legfontosabb metrikákkal és diagramokkal.

## Könyvtárstruktúra

A projekt futtatásához az alábbi mappa-struktúrának kell rendelkezésre állnia a gyökérkönyvtárban:

```text
Thesis/
├── Test_Data/        # Ide kell helyezni a nyers mérési adatokat (.xlsx formátumban). A fájlok első sorába kell a paramétereket feltüntetni. Egy minta található a feltöltött repozitóriumban.
├── Results/          # A generált ábrák, riportok (.html, .xlsx) és modellek (.pkl) helye
├── Cache/            # Ideiglenes fájlok a gyorsabb újrafutás érdekében
├── main.py           # A fő futtatható szkript
├── config.py         # Globális beállítások (könyvtárak, paraméterek, diagramstílusok)
├── utils.py          # Segédfüggvények (ábrázolás, adatkezelés, riportkészítés)
├── transformers.py   # Egyedi scikit-learn transzformátorok (VIF, interakciók, Scaler)
├── requirements.txt  # A projekt Python függőségei
├── Sample.xlsx       # Tribológiai adatsor minta
└── README.md         # Ez a dokumentum
```
*(Megjegyzés: A `Results` és a `Cache` mappákat a program automatikusan létrehozza, ha nem léteznek).*

## Telepítés

1. Győződjön meg róla, hogy Python 3.9 vagy újabb verzió van telepítve a számítógépére.
2. Nyisson egy parancssort a projekt mappájában, és telepítse a szükséges csomagokat:

```bash
pip install -r requirements.txt
```

## Használat

1. Helyezze a tribométeres méréseket tartalmazó Excel fájlokat (`.xlsx`) a `Test_Data` mappába. Az Excel fájloknak tartalmazniuk kell egy `Sheet Numeric SRA` (vagy elsődleges) munkalapot a megfelelő oszlopokkal (Time, Load, Temperature, COF stb.).
2. Futtassa a fő szkriptet:

```bash
python main.py
```

3. A futás befejeztével az eredmények megtalálhatók a `Results` mappában, és az összefoglaló `Eredmenyek_Riport.html` automatikusan megnyílik az alapértelmezett böngészőben.

### Gyorsítótár (Cache) használata
A program az adatok betöltését és a modellek betanítását alapértelmezetten gyorsítótárazza (cache) a gyorsabb fejlesztés és ábragenerálás érdekében. Ha a nyers adatok megváltoztak, vagy nulláról szeretné újratanítani a modelleket, állítsa a `config.py`-ban a `USE_CACHE` változót `False`-ra, vagy törölje a `Cache` mappa tartalmát.

## Kiegészítő szkriptek
A projekt tartalmaz néhány kiegészítő szkriptet is, amelyek a diplomamunkához szükséges extra ábrákat és folyamatábrákat generálják:
* `Hertzian_Stress.py`: Legenerálja a Hertz-féle feszültségeloszlást bemutató 3D-s ábrát a használt anyagpárosításról. Az anyagjellemzők megváltoztathatóak a kódban.
* `Stribeck_Curve.py`: Egy idealizált Stribeck-görbét rajzol ki.
* `3D_GIFS.py`: Mérési pontokat tartalmazó, ide-oda mozgó .gif fájlokat készít.

Ezeket a `python fájlnév.py` paranccsal külön lehet futtatni.

---
ENGLISH

# Machine learning pipeline for analyzing friction data

This project is a machine learning process designed to analyze and perform predictive modeling of tribometer measurement data, as well as to optimize lubrication conditions and friction coefficients. The code suggests supplementary, future measurements to ensure the best possible coverage of the measurement parameter space.

## Key Features
* **Data Processing:** Noise filtering (rolling mean), downsampling, and automatic removal of anomalies (outliers).
* **Feature Engineering:** Generation of physical parameters (e.g., Hertzian stress) and interaction features, VIF (Variance Inflation Factor)-based multicollinearity filtering.
* **Model Training and Optimization:** Hyperparameter tuning of 7 different algorithms (XGBoost, LightGBM, CatBoost, Random Forest, MLP, KNN, Ridge) using Optuna (Bayesian Optimization).
* **Ensemble:** Automatic selection and weighted combination of the top 3 models.
* **Explainability (XAI):** SHAP (SHapley Additive exPlanations)-based visualizations to understand the importance and impact of features.
* **Optimum search and DoE:** Determination of compromise (Pareto) optima, as well as suggestions for new measurement points (Active Learning / Sequential Design of Experiments).
* **Automatic reporting:** Generation of comprehensive HTML and Excel reports at the very end of the run, containing the most important metrics and charts.

## Directory Structure

To run the project, the following directory structure must be available in the root directory:

```text
Thesis/
├── Test_Data/        # Place the raw measurement data here (in .xlsx format). The parameters must be listed in the first row of the files. A sample is available in the uploaded repository.
├── Results/          # Location of generated figures, reports (.html, .xlsx), and models (.pkl)
├── Cache/            # Temporary files for faster reruns
├── main.py           # The main executable script
├── config.py         # Global settings (directories, parameters, plot styles)
├── utils.py          # Utility functions (plotting, data handling, report generation)
├── transformers.py   # Custom scikit-learn transformers (VIF, interactions, Scaler)
├── requirements.txt  # Python dependencies for the project
├── Sample.xlsx       # Tribological data table sample
└── README.md         # This document
```
*(Note: The `Results` and `Cache` folders are automatically created by the program if they do not exist).*

## Installation

1. Make sure Python 3.9 or a newer version is installed on your computer.
2. Open a command prompt in the project folder and install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Place the Excel files (`.xlsx`) containing the tribometer measurements in the `Test_Data` folder. The Excel files must contain a `Sheet Numeric SRA` (or primary) worksheet with the appropriate columns (Time, Load, Temperature, COF, etc.).
2. Run the main script:

```bash
python main.py
```

3. Once the run is complete, the results can be found in the `Results` folder, and the summary `Results_Report.html` will automatically open in your default browser.

### Using the Cache
By default, the program caches data loading and model training to speed up development and plot generation. If the raw data has changed, or if you want to retrain the models from scratch, set the `USE_CACHE` variable to `False` in `config.py`, or delete the contents of the `Cache` folder.

## Supplementary Scripts
The project also includes several supplementary scripts that generate the extra figures and flowcharts required for the thesis:
* `Hertzian_Stress.py`: Generates a 3D figure illustrating the Hertzian stress distribution for the used materials. The material properties can be modified in the code.
* `Stribeck_Curve.py`: Plots an idealized Stribeck curve for demonstration.
* `3D_GIFS.py`: Creates .gif files containing measurement points that move back and forth.

These can be run separately using the `python filename.py` command.
