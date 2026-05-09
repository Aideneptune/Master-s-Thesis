# Súrlódási adatok elemzésére készített gépi tanulási pipeline

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
* `Hertzian_Stress.py`: Legenerálja a Hertz-féle feszültségeloszlást bemutató 3D-s ábrát.
* `Stribeck_Curve.py`: Egy idealizált Stribeck-görbét rajzol ki.
* `3D_GIFS.py`: Mérési pontokat tartalmazó, ide-oda mozgó .gif fájlokat készít.

Ezeket a `python fájlnév.py` paranccsal külön lehet futtatni.
