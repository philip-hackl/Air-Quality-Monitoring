
# Luftqualitätsanalyse & Zukunftsprognose
Data driven Air Quality Monitoring: Predicting Health Risk Worldwide

## Ziel

Dieses Projekt bietet eine interaktive Streamlit-Anwendung zur Überwachung und Visualisierung der Luftqualität in Städten weltweit. Es nutzt die OpenAQ API, um Luftqualitätsdaten abzurufen, speichert diese in CSV-Dateien und ermöglicht die Visualisierung der Daten auf Karten und Diagrammen. Die Anwendung integriert Geocoding zur Bestimmung der Koordinaten von Städten und verwendet Folium für die Kartendarstellung und Plotly für die Diagrammerstellung.


## Inhaltsverzeichnis

1. [Überblick](#überblick)
2. [Features](#features)
3. [Installation](#installation)
4. [Anwendung](#anwendung)
5. [Wichtige Parameter](#wichtige-parameter)
6. [Anmerkungen](#anmerkungen)
7. [Fehlerbehebung](#fehlerbehebung)
8. [Lizenz](#lizenz)
9. [Links](#links)

## Features

- **Land- und Stadtauswahl:** Wählen Sie ein Land und eine Stadt aus und laden Sie die entsprechenden Luftqualitätsdaten.
- **Datenvisualisierung:** Erstellen Sie Zeitreihen-Diagramme und andere Visualisierungen zur Analyse der Luftqualität.
- **Datenaufbereitung für maschinelles Lernen:** Bereiten Sie die Daten für maschinelles Lernen vor, inklusive deskriptiver Statistiken und Korrelationsanalysen.
- **Zukunftsprognose:** Verwenden Sie lineare Regression zur Prognose der Luftqualität für die nächsten 30 Tage.

## Installation

1. **Systemvoraussetzungen:**
   - Python 3.7 oder höher
   - Streamlit
   - Pandas
   - NumPy
   - Matplotlib
   - Seaborn
   - Plotly
   - Scikit-Learn

2. **Installieren Sie die erforderlichen Pakete:**

   ```bash
   pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn folium
   ```

## Anwendung

### Konfiguration der Seite

Setzen Sie die Streamlit-Seite für ein breiteres Layout und konfigurieren Sie das Seaborn-Thema:

```python
st.set_page_config(layout="wide")
sns.set_theme(style="whitegrid")
```

### Sidebar-Optionen

- **Landkarte erstellen:** Aktiviert die Option zur Visualisierung der Stadt auf einer Landkarte.
- **Daten Vorbereitung für ML:** Bereitet die Daten für maschinelles Lernen vor.
- **Parameter Info:** Zeigt Erklärungen zu den Luftqualitätsparametern an.
- **CSV speichern aktivieren:** Speichert die Daten in einer CSV-Datei.
- **Debugging anzeigen:** Aktiviert Debugging-Optionen zur Fehlersuche.

### Datenabruf und -verarbeitung

- **Land und Stadt auswählen:** Wählen Sie ein Land und eine Stadt aus.
- **Zeitraum angeben:** Geben Sie den Zeitraum für die Datenabfrage an.
- **Daten abrufen und verarbeiten:** Überprüfen Sie, ob die Daten im Session State oder in einer CSV-Datei vorhanden sind, und laden Sie sie gegebenenfalls von der API.
- **Fehlerbehandlung und Zeitformatierung:** Führen Sie die Datenbereinigung und Formatierung durch.

### Visualisierungen

- **Zeitreihendiagramme:** Visualisieren Sie die durchschnittlichen Luftqualitätswerte.
- **Boxplots und Heatmaps:** Analysieren Sie die Daten mithilfe von Boxplots und Korrelationsheatmaps.
- **Prognosen:** Visualisieren Sie die Prognosen für die nächsten 30 Tage.

### Zukunftsprognose

- **Lineare Regression:** Verwenden Sie lineare Regression zur Prognose der Luftqualität.
- **Modellbewertung:** Bewerten Sie das Modell anhand von MAE, MSE und RMSE.
- **Visualisierung der Prognosen:** Zeigen Sie die Prognosen für die nächsten 30 Tage an.

## Wichtige Parameter

- **PM2.5:** Feinstaub kleiner als 2,5 Mikrometer.
- **PM10:** Feinstaub kleiner als 10 Mikrometer.
- **NO₂:** Stickstoffdioxid.
- **CO:** Kohlenmonoxid.
- **O₃:** Ozon.
- **SO₂:** Schwefeldioxid.
- **CH₄:** Methan.
- **BC:** Schwarzer Kohlenstoff.

## Anmerkungen

- **MAE (Mean Absolute Error):** Durchschnittlicher Fehler zwischen tatsächlichen und prognostizierten Werten.
- **MSE (Mean Squared Error):** Durchschnittlicher quadratischer Fehler; bestraft größere Fehler stärker.
- **RMSE (Root Mean Squared Error):** Quadratwurzel des MSE; bietet Fehlermaße in den gleichen Einheiten wie die Daten.

## Fehlerbehebung

- **Keine Länder oder Städte verfügbar:** Überprüfen Sie Ihre API-Verbindung oder wählen Sie ein anderes Land/Stadt.
- **Keine Daten gefunden:** Überprüfen Sie den Zeitraum und die Verfügbarkeit von Daten.

## Links
- [https://docs.openaq.org/docs/getting-started](https://docs.openaq.org/docs/getting-started)
- [https://python-graph-gallery.com/animation/](https://docs.openaq.org/reference/measurements_get_v2_measurements_get)
- [https://explore.openaq.org/#1.78/36.7/-18.7](https://explore.openaq.org/#1.78/36.7/-18.7)
