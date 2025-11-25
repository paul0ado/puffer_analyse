import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Titel der App
st.title("Geräte-Vergleichs-Analyse")

# 1. Datei-Upload in der Sidebar oder Main Area
uploaded_file = st.sidebar.file_uploader("Excel-Datei hochladen", type=["xlsx"])

# Funktion zum Laden der Daten (mit Caching für bessere Performance)
@st.cache_data
def load_data(file):
    df = pd.read_excel(file, sheet_name="Daten")
    # Bereinigung
    df['Messwert'] = pd.to_numeric(df['Messwert'], errors='coerce')
    # Wichtig: Spaltennamen bereinigen (Leerzeichen am Ende entfernen)
    df.columns = df.columns.str.strip()
    return df

# Statistik-Funktion für LinCCC
def linCCC(x, y):
    if len(x) < 2: return np.nan
    mean_lin_x = np.mean(x)
    mean_lin_y = np.mean(y)
    cov = np.cov(x, y, ddof=1)[0,1]
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    
    # Schutz vor Division durch Null
    denominator = var_x + var_y + (mean_lin_x - mean_lin_y)**2
    if denominator == 0:
        return np.nan
        
    ccc = (2*cov) / denominator
    return ccc

# Hauptlogik nur ausführen, wenn Datei da ist
if uploaded_file is not None:
    # Daten laden
    try:
        df = load_data(uploaded_file)
        st.success("Datei erfolgreich geladen!")
    except Exception as e:
        st.error(f"Fehler beim Laden der Datei: {e}")
        st.stop()

    # --- DATEN VORBEREITUNG ---
    # Filterung der Test-Gruppen
    df_AB = df.groupby("Testnummer").filter(lambda x: {"A", "B"}.issubset(set(x["Gerät"])))
    df_AC = df.groupby("Testnummer").filter(lambda x: {"A", "C"}.issubset(set(x["Gerät"])))
    
    anzahl_AB = df_AB["Testnummer"].nunique()
    anzahl_AC = df_AC["Testnummer"].nunique()

    # --- SIDEBAR STEUERUNG ---
    st.sidebar.header("Einstellungen")
    
    # Auswahl 1: Welcher Vergleich?
    vergleich_optionen = {
        "Gerät A vs. B": {"data": df_AB, "partner": "B", "count": anzahl_AB},
        "Gerät A vs. C": {"data": df_AC, "partner": "C", "count": anzahl_AC}
    }
    
    modus_label = st.sidebar.selectbox(
        "Welchen Geräte sollen verglichen werden?",
        options=list(vergleich_optionen.keys()),
        format_func=lambda x: f"{x} ({vergleich_optionen[x]['count']} Tests)"
    )
    
    # Setzen der Arbeitsvariablen basierend auf Auswahl
    auswahl_config = vergleich_optionen[modus_label]
    aktuelle_daten = auswahl_config["data"]
    vergleich_geraet = auswahl_config["partner"]

    # Auswahl 2: Welche Einheit?
    # Nur Einheiten anzeigen, die in den gefilterten Daten auch vorkommen
    verfuegbare_einheiten = sorted(aktuelle_daten['Einheit'].unique())
    
    if not verfuegbare_einheiten:
        st.warning("Keine Einheiten in den ausgewählten Daten gefunden.")
        st.stop()

    auswahl_einheit = st.sidebar.selectbox(
        "Einheit auswählen:",
        options=verfuegbare_einheiten
    )

    # --- HAUPT ANALYSE ---
    st.header(f"Analyse: {modus_label} | Einheit: {auswahl_einheit}")

    # Daten filtern für die gewählte Einheit
    # Wir nutzen hier die Schlüssel zum Mergen. 'Lotnummer' ohne Leerzeichen dank Bereinigung oben.
    cols = ["Testnummer", "Probe", "Lotnummer", "Messwert", "Gerät"]
    
    # Filtern
    df_filtered = aktuelle_daten.loc[aktuelle_daten["Einheit"] == auswahl_einheit, cols].copy()
    
    # Aufteilen
    df_A = df_filtered[df_filtered["Gerät"] == 'A']
    df_Partner = df_filtered[df_filtered["Gerät"] == vergleich_geraet]

    # Mergen
    df_merged = pd.merge(
        df_A, 
        df_Partner, 
        on=["Testnummer", "Probe", "Lotnummer"], 
        suffixes=('_A', f'_{vergleich_geraet}'),
        how='inner'
    )
    
    col_A = "Messwert_A"
    col_Partner = f"Messwert_{vergleich_geraet}"

    # NaNs entfernen
    df_merged = df_merged.dropna(subset=[col_A, col_Partner])
    
    anzahl_paare = len(df_merged)
    st.write(f"Anzahl gültiger Datenpaare: **{anzahl_paare}**")

    if anzahl_paare > 1:
        x = df_merged[col_A].values
        y = df_merged[col_Partner].values
        
        # --- STATISTIK BERECHNUNG ---
        var_x = np.var(x, ddof=1)
        var_y = np.var(y, ddof=1)
        
        # F-Test Logik
        f_pruf = 0
        f_krit = 0
        h0_text = "-"
        groesser = "-"
        
        if var_x > 0 and var_y > 0:
            alpha = 0.05
            df1 = len(x) - 1
            df2 = len(y) - 1
            f_krit = stats.f.ppf(1 - alpha, df1, df2)
            
            if var_x > var_y:
                f_pruf = var_x / var_y
                groesser = "Gerät A (x)"
            elif var_y > var_x:
                f_pruf = var_y / var_x
                groesser = f"Gerät {vergleich_geraet} (y)"
            else:
                f_pruf = 1
                groesser = "gleich"

            if f_pruf > f_krit:
                h0_text = "H0 verworfen (Varianzen unterschiedlich)"
            else:
                h0_text = "H0 nicht verworfen (Varianzen gleich)"
        else:
            h0_text = "Varianz ist 0, Test nicht möglich"

        ccc_wert = linCCC(x, y)

        # --- ERGEBNISSE ANZEIGEN ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Statistische Kennzahlen")
            st.write(f"Varianz A: `{var_x:.5f}`")
            st.write(f"Varianz {vergleich_geraet}: `{var_y:.5f}`")
            st.write(f"LinCCC: `{ccc_wert:.4f}`" if ccc_wert is not None else "LinCCC: -")
        
        with col2:
            st.subheader("F-Test Ergebnisse")
            st.write(f"Größere Varianz: {groesser}")
            st.write(f"F-Prüfgröße: `{f_pruf:.3f}`")
            st.write(f"F-Kritisch: `{f_krit:.3f}`")
            if "verworfen" in h0_text:
                st.success(h0_text)
            else:
                st.error(h0_text)

        # --- PLOT ---
        st.subheader("Streudiagramm")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter Plot
        ax.scatter(x, y, alpha=0.7, label='Messwerte')
        
        # Ideallinie (y=x)
        # Wir nehmen min/max von beiden Achsen für eine schöne Diagonale
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, ls = "-", color = "gray", alpha=0.75, label='Ideallinie (x=y)')
        
        ax.set_xlabel("Gerät A")
        ax.set_ylabel(f"Gerät {vergleich_geraet}")
        ax.set_title(f"Vergleich A vs {vergleich_geraet} ({auswahl_einheit})")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        st.pyplot(fig)
        
        # Optional: Daten anzeigen
        with st.expander("Rohdaten anzeigen"):
            st.dataframe(df_merged)

    else:
        st.warning("Zu wenige Datenpunkte für eine Analyse (weniger als 2 Paare).")

else:
    st.info("Bitte laden Sie eine Excel-Datei hoch, um zu beginnen.")
