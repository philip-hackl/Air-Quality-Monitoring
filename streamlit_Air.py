

import streamlit as st
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import datetime
import time
import plotly.express as px
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import folium
from streamlit_folium import folium_static


# 1 - OpenAQ API Klasse
class OpenAQ:
    BASE_URL = 'https://api.openaq.org/v2/'
    API_CODE = "8e0694ed4e6ccd310176923774548c6b212dbb34f6eb71096f1ee8bdb8a105d8"

    def _make_request(self, endpoint, params=None):
        url = f"{self.BASE_URL}{endpoint}"
        headers = {"X-API-Key": self.API_CODE}  # API-Schlüssel im Header
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 429:  # Rate limit exceeded
                st.warning("Rate limit exceeded. Waiting before retrying...")
                time.sleep(60)  # Wait for 60 seconds
                response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Error during API request: {e}")
            return None

    def get_countries(self):
        data = self._make_request('countries')
        return data['results'] if data else []

    def get_cities(self, country=None):
        params = {'country': country} if country else {}
        data = self._make_request('cities', params)
        return data['results'] if data else []

    def get_measurements(self, city=None, country=None, date_from=None, date_to=None, limit=1000, parameter=None):
        params = {
            'city': city,
            'country': country,
            'date_from': date_from,
            'date_to': date_to,
            'limit': limit
        }
        if parameter:
            params['parameter'] = parameter
        data = self._make_request('measurements', params)
        return data['results'] if data else []




# Funktion zum Speichern von Daten in eine CSV-Datei
def save_to_csv(df, filename):
    df.to_csv(filename, index=False)

# Funktion zum Überprüfen, ob die CSV-Datei bereits vorhanden ist
def is_csv_present(filename):
    return os.path.exists(filename)

# Funktion zum Laden von Daten aus einer CSV-Datei
def load_from_csv(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        st.error("CSV-Datei nicht gefunden.")
        return None

# Stelle sicher, dass die Session State initialisiert ist
if 'city_data' not in st.session_state:
    st.session_state['city_data'] = {}

# Erweiterung zur Speicherung von Daten in der Session State
def store_city_data(city, data):
    st.session_state['city_data'][city] = data

# Erweiterung zum Abrufen gespeicherter Daten aus der Session State
def load_city_data(city):
    return st.session_state['city_data'].get(city, None)



# Funktion, um die Koordinaten einer Stadt zu bekommen
def get_city_coordinates(city_name):
    geolocator = Nominatim(user_agent="AirQuality_Projekt")  # Füge einen User-Agent hinzu
    try:
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude
        else:
            st.error(f"Keine Koordinaten gefunden für {city_name}.")
            return None, None
    except GeocoderTimedOut:
        st.error("Die Geocoding-Anfrage hat zu lange gedauert.")
        return None, None
    except Exception as e:
        st.error(f"Fehler bei der Geocoding-Anfrage: {str(e)}")
        return None, None


# Funktion zum Erstellen und Anzeigen einer Folium-Karte
def create_folium_map(city_name):
    # Holen der Koordinaten für die angegebene Stadt
    latitude, longitude = get_city_coordinates(city_name)
    
    if latitude is not None and longitude is not None:
        # Erstelle eine Folium-Karte
        m = folium.Map(location=[latitude, longitude], zoom_start=12)

        # Füge einen Marker für die Stadt hinzu
        folium.Marker(
            location=[latitude, longitude], 
            popup=f"{city_name}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
        
        # Füge einen Kreis-Marker hinzu
        folium.CircleMarker(
            location=[latitude, longitude], 
            radius=10, 
            color='red', 
            fill=True, 
            fill_color='red'
        ).add_to(m)

        # Beispiel-Linien-Marker: Hier kannst du reale Koordinaten einfügen
        example_lat_lons = [
            [latitude, longitude],
            [latitude + 0.01, longitude + 0.01]  # Beispiel-Koordinaten für die Linie
        ]
        folium.PolyLine(
            locations=example_lat_lons, 
            color='green'
        ).add_to(m)

        # Zeige die Karte in Streamlit an
        folium_static(m)
    else:
        st.error(f"Koordinaten für {city_name} konnten nicht gefunden werden.")



# Fallback-Mechanismus zur Datenabfrage
def get_measurements_with_fallback(api, city, country_code, date_from, date_to, parameter, attempts=3):
    for attempt in range(attempts):
        try:
            st.write(f"Versuch {attempt + 1}: Abrufen der Daten von {date_from} bis {date_to}")
            measurements = api.get_measurements(city=city, country=country_code, date_from=date_from, date_to=date_to, parameter=parameter)
            if measurements:
                return measurements
            else:
                # Erweitere den Zeitraum und versuche es erneut
                date_from = (pd.to_datetime(date_from) - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
                date_to = (pd.to_datetime(date_to) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        except Exception as e:
            st.error(f"Fehler beim Abrufen der Daten: {e}")
            break
    st.write("Keine Daten gefunden nach mehreren Versuchen.")
    return None


# Verfügbarkeitsprüfung einfügen (optional)
def check_data_availability(api, city, country_code, parameter, date_from, date_to):
    try:
        measurements = api.get_measurements(city=city, country=country_code, date_from=date_from, date_to=date_to, parameter=parameter, limit=1)
        return len(measurements) > 0
    except Exception as e:
        st.error(f"Fehler bei der Überprüfung der Datenverfügbarkeit: {e}")
        return False


# Daten abrufen und in einem DataFrame kombinieren
# Funktion zum Abrufen und Kombinieren von Messdaten für verschiedene Parameter
def get_all_parameters_data(api, city, country_code, date_from, date_to):
    # Liste der Parameter, für die Messdaten abgerufen werden sollen
    parameters = ["pm25", "pm10", "no2", "so2", "o3", "co", "bc"]
    # Leere Liste zur Speicherung der DataFrames für jeden Parameter
    dfs = []

    # Schleife über jeden Parameter, um die Messdaten abzurufen
    for parameter in parameters:
        st.write(f"Abrufen der Daten für {parameter}")  # Ausgabe einer Nachricht, dass Daten für den aktuellen Parameter abgerufen werden
        # Abrufen der Messdaten vom API für den aktuellen Parameter
        measurements = api.get_measurements(city=city, country=country_code, date_from=date_from, date_to=date_to, parameter=parameter)
        # Überprüfen, ob Messdaten zurückgegeben wurden
        if measurements:
            # Erstellen eines DataFrames aus den abgerufenen Messdaten
            df = pd.DataFrame(measurements)
            st.dataframe(df)  # Zeigt den DataFrame nach dem Laden an, um den Inhalt zu überprüfen
            # Umwandeln der 'date' Spalte in Datetime-Format und Extrahieren des UTC-Zeitstempels
            df['date'] = pd.to_datetime(df['date'].apply(lambda x: x['utc']))
            # Auswahl relevanter Spalten
            df = df[['date', 'location', 'parameter', 'value', 'unit']]
            # Pivotieren des DataFrames, um die Parameter in Spalten zu transformieren, mit Mittelwert als Aggregationsfunktion
            dfs.append(df.pivot_table(index=['date', 'location'], columns='parameter', values='value', aggfunc='mean').reset_index())
        else:
            st.write(f"Keine Daten für {parameter} gefunden.")  # Nachricht, wenn keine Daten für den aktuellen Parameter gefunden wurden
    
    # Überprüfen, ob DataFrames gesammelt wurden
    if dfs:
        # Kombinieren aller DataFrames in der Liste entlang der Zeilenachse (axis=0) und Aggregieren nach Datum und Standort
        combined_df = pd.concat(dfs, axis=0).groupby(['date', 'location']).first().reset_index()
        return combined_df  # Rückgabe des kombinierten DataFrames

    # Dieser Block wird nur ausgeführt, wenn keine Daten für die Parameter abgerufen wurden
    if dfs:
        combined_df = pd.concat(dfs, axis=0)  # Kombinieren der DataFrames

        # Entfernen von Spalten, die in einigen DataFrames fehlen könnten
        all_columns = set(combined_df.columns)
        valid_parameters = [param for param in parameters if param in all_columns]

        # Entfernen von Zeilen, in denen alle Parameter NaN sind
        combined_df = combined_df.dropna(subset=valid_parameters, how='all')

        # Aggregieren nach Datum und Location, wobei nur die erste Zeile behalten wird
        combined_df = combined_df.groupby(['date', 'location'], as_index=False).first()

        # Entfernen aller Zeilen mit NaN-Werten in den Parametern
        combined_df = combined_df.dropna(subset=valid_parameters)

        return combined_df  # Rückgabe des finalen kombinierten DataFrames
    else:
        st.write("Keine Daten gefunden.")  # Nachricht, wenn keine Daten vorhanden sind
        return None  # Rückgabe von None, wenn keine Daten verfügbar sind



def plot_air_quality_with_who_limits(df):
    # PM2.5
    if 'pm25' in df.columns:
        fig = px.line(df, x=df.index, y='pm25', labels={'value': 'PM2.5 (µg/m³)', 'index': 'Datum'}, title="Durchschnittswerte der Luftqualität: PM2.5 - Feinstaub kleiner als 2,5 Mikrometer")
        fig.add_hline(y=25, line_dash="dash", line_color=fig.data[0].line.color, annotation_text="WHO 24-Stunden-Grenzwert (25 µg/m³)", annotation_position="top left")
        fig.add_hline(y=10, line_dash="dash", line_color=fig.data[0].line.color, annotation_text="WHO Jahresmittelwert (10 µg/m³)", annotation_position="bottom left")
        st.plotly_chart(fig)

    # PM10
    if 'pm10' in df.columns:
        fig = px.line(df, x=df.index, y='pm10', labels={'value': 'PM10 (µg/m³)', 'index': 'Datum'}, title="Durchschnittswerte der Luftqualität: M10 - Feinstaub kleiner als 10 Mikrometer")
        fig.add_hline(y=50, line_dash="dash", line_color=fig.data[0].line.color, annotation_text="WHO 24-Stunden-Grenzwert (50 µg/m³)", annotation_position="top left")
        fig.add_hline(y=20, line_dash="dash", line_color=fig.data[0].line.color, annotation_text="WHO Jahresmittelwert (20 µg/m³)", annotation_position="bottom left")
        st.plotly_chart(fig)

    # NO₂
    if 'no2' in df.columns:
        fig = px.line(df, x=df.index, y='no2', labels={'value': 'NO₂ (µg/m³)', 'index': 'Datum'}, title="Durchschnittswerte der Luftqualität: NO₂ - Stickstoffdioxid")
        fig.add_hline(y=200, line_dash="dash", line_color=fig.data[0].line.color, annotation_text="WHO 1-Stunden-Grenzwert (200 µg/m³)", annotation_position="top left")
        fig.add_hline(y=40, line_dash="dash", line_color=fig.data[0].line.color, annotation_text="WHO Jahresmittelwert (40 µg/m³)", annotation_position="bottom left")
        st.plotly_chart(fig)

    # O₃
    if 'o3' in df.columns:
        fig = px.line(df, x=df.index, y='o3', labels={'value': 'O₃ (µg/m³)', 'index': 'Datum'}, title="Durchschnittswerte der Luftqualität: O₃ - Ozon")
        fig.add_hline(y=100, line_dash="dash", line_color=fig.data[0].line.color, annotation_text="WHO 8-Stunden-Grenzwert (100 µg/m³)", annotation_position="top left")
        st.plotly_chart(fig)

    # SO₂
    if 'so2' in df.columns:
        fig = px.line(df, x=df.index, y='so2', labels={'value': 'SO₂ (µg/m³)', 'index': 'Datum'}, title="Durchschnittswerte der Luftqualität: SO₂ - Schwefeldioxid")
        fig.add_hline(y=20, line_dash="dash", line_color=fig.data[0].line.color, annotation_text="WHO Tages-Grenzwert (20 µg/m³)", annotation_position="top left")
        st.plotly_chart(fig)

    # CO
    if 'co' in df.columns:
        fig = px.line(df, x=df.index, y='co', labels={'value': 'CO (mg/m³)', 'index': 'Datum'}, title="Durchschnittswerte der Luftqualität: CO - Kohlenmonoxid")
        fig.add_hline(y=10, line_dash="dash", line_color=fig.data[0].line.color, annotation_text="WHO 1-Stunden-Grenzwert (10 mg/m³)", annotation_position="top left")
        fig.add_hline(y=7, line_dash="dash", line_color=fig.data[0].line.color, annotation_text="WHO 8-Stunden-Grenzwert (7 mg/m³)", annotation_position="bottom left")
        st.plotly_chart(fig)






###########

def main():
    # Konfiguration der Streamlit-Seite für ein breiteres Layout
    st.set_page_config(layout="wide")

    # Setzen des Seaborn-Themas für Plots
    sns.set_theme(style="whitegrid")

    st.title("Luftqualitätsanalyse & Zukunftsprognose")

    api = OpenAQ()

    # Sidebar
    st.sidebar.header("Optionen")
    create_map = st.sidebar.checkbox("Landkarte erstellen", help= "Zeigt eine Landkarte der Stadt mit folium erstellt.", value= True)

    data_preperation = st.sidebar.checkbox("Daten Vorbereitung für ML", value= True)

    # Checkbox für Parameterinfo
    param_box = st.sidebar.checkbox("Parameter Info", help= "Zeigt eine Erklärung zu allen Parametern an.")

    # CSV speichern aktivieren/deaktivieren
    st.sidebar.subheader("Speichern")
    st.sidebar.write("Alle Suchanfragen werden automatisch in Streamlit 'Session_State' gespeichert.")
    save_to_csv_enabled = st.sidebar.checkbox("CSV speichern aktivieren", value=False)

    # Debugging Boxen
    st.sidebar.subheader("Debugging")

    # show_df1 = st.sidebar.checkbox("Zeige erhaltene Daten", help= "Zeige erhaltenen Daten als DataFrame. Die Direkten Daten der API nach dem Abruf.")

    loading_data = st.sidebar.checkbox("Geladene Daten zeigen", help= "Original geladene Daten anzeigen. (Combined DF nach dem Laden, wenn er vorhanden ist.)")

    debugging = st.sidebar.checkbox("Debugging anzeigen", help= "Zeigt die zwischen Schritte in der Durchführung an, um Fehler zu beheben.")



    # Anzeigen der Parameterinfo, wenn die Checkbox aktiviert ist
    if param_box:
        st.sidebar.markdown(""" 
        - - - - -           
        **Parameter Erklärung:**

        - **PM2.5 - Feinstaub kleiner als 2,5 Mikrometer:** Misst die Massenkonzentration (µg/m³) und Anzahl der Partikel (Partikel/cm³) dieser sehr feinen Partikel, die tief in die Lungen eindringen können.

        - **PM4 - Feinstaub kleiner als 4 Mikrometer:** Gibt die Massenkonzentration (µg/m³) und Anzahl der Partikel (Partikel/cm³) von etwas größeren Feinstaubpartikeln an, die immer noch gesundheitsschädlich sein können.

        - **PM10 - Feinstaub kleiner als 10 Mikrometer:** Erfasst die Massenkonzentration (µg/m³) und Anzahl der Partikel (Partikel/cm³) von gröberem Feinstaub, der die Atemwege erreichen kann.

        - **NO - Stickstoffmonoxid:** Misst die Konzentration von Stickstoffmonoxid (ppm), einem gasförmigen Schadstoff, der in urbanen Umgebungen häufig vorkommt.

        - **NO₂ - Stickstoffdioxid:** Erfasst die Konzentration von Stickstoffdioxid (ppm), das zur Bildung von Ozon und saurem Regen beitragen kann.

        - **CH₄ - Methan:** Gibt die Konzentration von Methan (ppm) an, einem Treibhausgas, das zur globalen Erwärmung beiträgt.

        - **SO₂ - Schwefeldioxid:** Misst die Konzentration von Schwefeldioxid (ppm), einem gasförmigen Schadstoff, der sauren Regen verursacht und Atemprobleme hervorrufen kann.

        - **O₃ - Ozon:** Erfasst die Konzentration von Ozon (ppm) in der Atmosphäre, das in der oberen Atmosphäre schützt, aber am Boden gesundheitsschädlich sein kann.

        - **CO - Kohlenmonoxid:** Gibt die Konzentration von Kohlenmonoxid (ppm) an, einem giftigen Gas, das bei unvollständiger Verbrennung entsteht.

        - **BC - Schwarzer Kohlenstoff):** Misst die Konzentration von schwarzem Kohlenstoff (µg/m³), der aus unverbrannten Kohlenwasserstoffen resultiert und zur Luftverschmutzung beiträgt.
        
        **Stunden-Grenzwerte:** Schutz vor den Auswirkungen von kurzzeitigen Spitzenbelastungen (hohe Konzentration über kurze Zeit).
        
        **Jahres-Grenzwerte:** Schutz vor den Auswirkungen einer langfristigen Exposition (niedrigere Konzentration über lange Zeit).
                            
        """)


    # Auswahl des Landes
    countries = api.get_countries()
    if countries:
        country_list = [f"{country.get('code', 'Unknown')} - {country.get('name', 'Unknown')}" for country in countries]
        default_country = "DE - Germany"
        selected_country = st.selectbox(
            "Wähle ein Land",
            country_list,
            index=country_list.index(default_country) if default_country in country_list else 0
        )
        country_code = selected_country.split(' - ')[0]
    else:
        st.error("Keine Länder verfügbar.")
        return

    # Auswahl der Stadt
    cities = api.get_cities(country=country_code)
    if cities:
        city_list = [city.get('city', 'Unknown') for city in cities]
        default_city = "Berlin"
        city = st.selectbox(
            "Wähle eine Stadt",
            city_list,
            index=city_list.index(default_city) if default_city in city_list else 0
        )
    else:
        st.error("Keine Städte in diesem Land gefunden.")
        return

    # Datumsauswahl
    date_from = st.date_input("Von Datum", value=pd.to_datetime('2023-01-01')).strftime('%Y-%m-%d')
    date_to = st.date_input("Bis Datum", value=pd.to_datetime('2023-12-31')).strftime('%Y-%m-%d')

    # CSV-Dateiname erstellen
    csv_filename = f"air_quality_{city}_{country_code}_{date_from}_{date_to}.csv"

    # Abrufen der Daten
    if st.button("Daten abrufen"):
        if create_map:
            create_folium_map(city)

        st.write(f"Daten für {city}, {selected_country} von {date_from} bis {date_to}")

        # Überprüfen, ob die Daten bereits in der Session State vorhanden sind
        combined_df = load_city_data(city)

        if combined_df is None:
            # Wenn Daten nicht im Session State vorhanden sind, überprüfen, ob die Daten in einer CSV-Datei gespeichert sind
            if is_csv_present(csv_filename):
                st.write("Lade Daten aus CSV-Datei...")
                combined_df = load_from_csv(csv_filename)
                st.write(f"Die Daten wurden aus der CSV-Datei '{csv_filename}' geladen.")
                # Speichern in der Session State für zukünftige Abfragen
                store_city_data(city, combined_df)
            else:
                # Wenn die Daten weder im Session State noch in der CSV-Datei vorhanden sind, von der API abrufen
                st.write("Daten werden von der API abgerufen...")
                combined_df = get_all_parameters_data(api, city, country_code, date_from, date_to)
                if combined_df is not None:
                    # Speichern in der Session State für zukünftige Abfragen
                    store_city_data(city, combined_df)

                    if save_to_csv_enabled:
                        save_to_csv(combined_df, csv_filename) 
                        st.write(f"Die Daten wurden in der CSV-Datei '{csv_filename}' gespeichert.")
                else:
                    st.write("Keine Daten gefunden.")
                    return

        # Falls Daten vorhanden sind, führe die Fehlerbehandlung und Konvertierung durch
        if combined_df is not None:
            st.header("Daten im Überblick:")

            if loading_data:
                st.markdown("**Original geladene Daten:**")
                st.dataframe(combined_df)

            # Füge die 'location'-Spalte hinzu und setze sie auf die gesuchte Stadt.
            df_together = combined_df.copy()
            df_together['location'] = city

            if debugging:
                st.markdown("**Debugging:** 'Location' wird zur gesuchten Stadt geändert (einheitlicher Ort)")
                st.dataframe(df_together)


            if 'date' in df_together.columns:
                # Konvertiere die 'date'-Spalte in das richtige Datetime-Format
                df_together['date'] = pd.to_datetime(df_together['date'], errors='coerce')

                if debugging:
                    st.markdown("**Debugging:** 'Date' Format prüfen - pd.to_datetime()")
                    st.dataframe(df_together)

                # Entferne Zeilen mit NaN in der 'date'-Spalte
                df_together = df_together.dropna(subset=['date'])

                if debugging:
                    st.markdown("**Debugging:** Fertig - Bereinigte Daten:")
                    st.dataframe(df_together)

                # Nur numerische Spalten für die Aggregation auswählen
                numeric_together_df = df_together.select_dtypes(include=[np.number])

                # Gruppieren nach 'datetime' und Aggregation durchführen
                aggregated_df = df_together.groupby('date', as_index= False).agg({col: 'mean' for col in numeric_together_df.columns})

                st.subheader(f"Zusammengefügte Daten für {city}: (Stündlich)")
                st.dataframe(aggregated_df)


                # Stelle sicher, dass nur numerische Spalten für die Resampling-Operation verwendet werden
                numeric_df = df_together.select_dtypes(include=[float, int]).copy()

                if debugging:
                    st.markdown("**Debugging:** Prüfen ob nur Numerische Werte vorhanden sind")  # Test 1
                    st.dataframe(numeric_df)  # Test 1

                if not numeric_df.empty:
                    # Setze die 'date'-Spalte als Index und resample die Daten
                    numeric_df.set_index(df_together['date'], inplace=True)

                    if debugging:
                        st.markdown("**Debugging:** Setzt 'date' als Index")  # Test 1
                        st.dataframe(numeric_df)  # Test 1

                    resampled_df = numeric_df.resample('D').mean().fillna(0)
                    
                    if debugging:
                        st.markdown("**Debugging:** Resample der Daten - von Stunden auf Tage - \n Vorbereitung für Machine Learning")  # Test 1
                        st.dataframe(resampled_df)  # Test 1


                    st.subheader("Visualisierungen:")

                    # Zeige den Plot der durchschnittlichen Luftqualitätswerte
                    st.write(px.line(resampled_df, title='Überblick der Durchschnittswerte der Luftqualität'))
                    # Visualisierung der Luftqualität mit WHO-Grenzwerten
                    plot_air_quality_with_who_limits(resampled_df)
                else:
                    st.write("Keine numerischen Daten zur Analyse vorhanden.")

            else:
                st.write("Die 'date'-Spalte ist nicht vorhanden oder leer.")


            # Zukunftsprognose
            st.header("Zukunftsprognose:")
            st.markdown("**Lineare Regression**")
            df_grouped = combined_df.groupby('date').mean(numeric_only=True).reset_index()
            # Konvertiere das Datum in eine numerische Form
            df_grouped['date'] = pd.to_datetime(df_grouped['date'])
            X = np.array((df_grouped['date'] - df_grouped['date'].min()).dt.days).reshape(-1, 1)

            if data_preperation:
                st.markdown("Daten Vorbereitung für ML", help= "Zeigt alle Daten in Nummerischer Form an. Auch Datum und Uhrzeit.")
                st.dataframe(df_grouped)

                # Zeigt die deskriptiven Statistiken des DataFrames an
                # Berechnung der deskriptiven Statistiken
                descriptive_stats = df_grouped.describe()
                # Transponieren der deskriptiven Statistiken
                descriptive_stats_transposed = descriptive_stats.transpose()
                # Anzeigen der transponierten deskriptiven Statistiken
                st.write("Überblick der Datenverteilung: (df.describe())")
                st.write(descriptive_stats_transposed)

                # Korrelationsmatrix berechnen
                corr = df_grouped.corr()
                # Heatmap der Korrelationsmatrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
                plt.title('Korrelationsmatrix')
                # Zeige den Plot in Streamlit an
                st.pyplot(plt)

                plt.figure(figsize=(12, 8))
                # Boxplots für jede Messgröße
                sns.boxplot(data=df_grouped)
                plt.xticks(rotation=90)
                plt.title('Boxplots der Messgrößen')
                # Zeige den Plot in Streamlit an
                st.pyplot(plt)

            # Annahme: df_grouped ist bereits definiert und enthält die notwendigen Daten
            st.subheader("Prognose für die nächsten 30 Tage:")
            future_df = pd.DataFrame()

            # Liste zur Speicherung der Evaluierungsergebnisse
            evaluation_results = []

            # Durchlaufe jede zu prognostizierende Variable
            for param in ["pm25", "pm10", "no2", "so2", "o3", "co", "bc"]:
                if param in df_grouped.columns:
                    # Bereite die Daten vor
                    y = df_grouped[param].fillna(0)
                    X = (df_grouped['date'] - df_grouped['date'].min()).dt.days.values.reshape(-1, 1)  # Tage seit dem Startdatum

                    # Teile die Daten in Trainings- und Testdatensatz auf
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Trainiere das Modell
                    model = LinearRegression()
                    model.fit(X_train, y_train)

                    # Prognostiziere zukünftige Daten
                    future_dates = np.array([(df_grouped['date'].max() + pd.Timedelta(days=i)).date() for i in range(1, 31)])
                    future_days = np.array((future_dates - df_grouped['date'].min().date()).astype('timedelta64[D]').astype(int)).reshape(-1, 1)
                    future_predictions = model.predict(future_days)
                    future_df[param] = pd.Series(future_predictions, index=future_dates)

                    # Evaluierung auf Testdaten
                    pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, pred)
                    mse = mean_squared_error(y_test, pred)
                    rmse = np.sqrt(mse)

                    # Speichere die Ergebnisse in der Liste
                    evaluation_results.append({
                        "Parameter": param,
                        "Mean Absolute Error (MAE) %": mae,
                        "Mean Squared Error (MSE) %": mse,
                        "Root Mean Squared Error (RMSE) %": rmse
                    })

            # Erstelle einen DataFrame aus den Evaluierungsergebnissen
            evaluation_df = pd.DataFrame(evaluation_results)

            # Zeige die Evaluierungsergebnisse an
            st.write("Evaluierung der Modelle:")
            st.dataframe(evaluation_df)

            st.info("Mean Absolute Error (MAE): MAE misst den durchschnittlichen Fehler zwischen den tatsächlichen und den vorhergesagten Werten in den gleichen Einheiten wie die Daten. Ein niedriger MAE bedeutet, dass die Vorhersagen im Durchschnitt nah an den tatsächlichen Werten liegen.")
                    
            st.info("Mean Squared Error (MSE): MSE berechnet den durchschnittlichen quadratischen Fehler zwischen den tatsächlichen und den vorhergesagten Werten, wobei größere Fehler stärker gewichtet werden. Ein niedriger MSE zeigt an, dass das Modell generell bessere Vorhersagen trifft, indem es große Fehler stärker bestraft.")
            
            st.info("Root Mean Squared Error (RMSE): RMSE ist die Quadratwurzel des MSE und bringt den Fehler auf die gleiche Skala wie die Originaldaten zurück. Es bietet eine verständliche Metrik für den durchschnittlichen Fehler in den gleichen Einheiten wie die Daten, wobei größere Fehler besonders betont werden.")
            
            if not future_df.empty:
                st.subheader("Zukünftige Vorhersagen:")
                st.dataframe(future_df)
                st.markdown("**Visualisierung der Prognose für die nächsten 30 Tage:**")
                st.line_chart(future_df)
            else:
                st.write("Keine zukünftigen Vorhersagen verfügbar.")
    else:
        st.write("Keine Daten gefunden. Versuche einen anderen Zeitraum oder eine andere Stadt.")







if __name__ == "__main__":
    main()
