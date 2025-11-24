import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# --- Titel der Webapp ---
st.set_page_config(page_title="Excel Analyse Tool", layout="centered")
st.title("📊 Statistische Auswertung & Analyse")
st.markdown("Lade deine Excel-Datei hoch, um die Analyse für **Chargen** oder **PK** zu starten.")

# --- 1. Datei Upload & Input ---
with st.sidebar:
    st.header("Einstellungen")
    uploaded_file = st.file_uploader("Excel Datei hochladen", type="xlsx")
    st.divider()
    auswahl = st.radio("Welches Set analysieren?", ("Chargen", "PK"))

# --- 2. Daten laden ---
if uploaded_file is not None:
    try:
        # Blatt-Auswahl Logik beibehalten
        sheet_index = 0 if auswahl == "Chargen" else 1
        
        df_aktuell = pd.read_excel(uploaded_file, sheet_name=sheet_index)
        
        # Kurzer Check, ob die Spalten existieren
        if "ZMB" not in df_aktuell.columns or "INF" not in df_aktuell.columns:
            st.error(f"Fehler: Die Spalten 'ZMB' und 'INF' wurden im Blatt '{auswahl}' nicht gefunden.")
            st.stop()
            
        df_aktuell = df_aktuell.dropna()
        spalte_ZMB = df_aktuell["ZMB"]
        
        # --- 3. Berechnungen (Logik 1:1 übernommen) ---
        messungen = np.arange(len(spalte_ZMB))
        
        # Barplot Vorbereitung
        x_bar = np.arange(len(messungen))
        breite = 0.35
        position_ZMB = x_bar - breite/2
        position_INF = x_bar + breite/2

        # Regression
        m, b = np.polyfit(df_aktuell["ZMB"], df_aktuell["INF"], 1)
        regression_line_ZMB = m * df_aktuell["ZMB"] + b
        ZMB_INF_pearson_matrix = np.corrcoef(df_aktuell["ZMB"], df_aktuell["INF"])
        ZMB_INF_pearson = np.round(ZMB_INF_pearson_matrix[0,1], 3)

        # Lin CCC Funktion
        def linCCC(ZMB, INF):
            mean_lin_ZMB = np.mean(ZMB)
            mean_lin_INF = np.mean(INF)
            cov = np.cov(ZMB, INF, ddof=1)[0,1]
            var_ZMB = np.var(ZMB, ddof=1)
            var_INF = np.var(INF, ddof=1)
            ccc = (2*cov)/(var_ZMB + var_INF + (mean_lin_ZMB - mean_lin_INF)**2)
            return ccc

        # Bland-Altman
        mittelwerte = (df_aktuell["ZMB"] + df_aktuell["INF"]) / 2
        diff = df_aktuell["ZMB"] - df_aktuell["INF"]
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff

        # Stats Auswertung
        result = stats.ttest_rel(df_aktuell["ZMB"], df_aktuell["INF"])
        ci = result.confidence_interval(confidence_level=0.95)
        mean_INF = np.mean(df_aktuell["INF"])
        aeq_upper = 0.25 * mean_INF
        aeq_lower = - 0.20 * mean_INF
        ist_aequivalent = (ci.low > aeq_lower) and (ci.high < aeq_upper)

        # --- 4. Darstellung der Ergebnisse ---
        
        st.divider()
        st.subheader("Statistische Kennzahlen")
        
        # Metriken schön in Spalten anzeigen statt print()
        m1, m2, m3 = st.columns(3)
        m1.metric("Anzahl Messungen (n)", len(spalte_ZMB))
        m2.metric("Mittlerer INF-Gehalt", f"{mean_INF:.2f}")
        m3.metric("Bias (Diff ZMB-INF)", f"{mean_diff:.3f}")

        st.caption(f"Standardabweichung der Differenzen (SD): {std_diff:.3f}")
        st.caption(f"95% CI des Bias: [{ci.low:.3f} bis {ci.high:.3f}]")
        
        # Äquivalenz Check mit farbigen Boxen
        st.write(f"**Äquivalenzgrenzen (80-125%):** {aeq_lower:.3f} bis +{aeq_upper:.3f}")
        
        if ist_aequivalent:
            st.success("✅ Äquivalenz bestätigt: Das 95% CI des Bias liegt innerhalb der Grenzen.")
        else:
            st.error("❌ Äquivalenz NICHT bestätigt: Das 95% CI überschreitet die Grenzen.")

        # --- 5. Plots ---
        st.subheader("Grafische Auswertung")
        
        # Figure Setup
        fig, ax = plt.subplots(3, 1, figsize=(6, 14))
        
        # Plot 1: Barplot
        ax[0].bar(position_ZMB, df_aktuell["ZMB"], breite, color="#297fc1", zorder=3, label="ZMB")
        ax[0].bar(position_INF, df_aktuell["INF"], breite, color="#ffb64e", zorder=3, label="INF")
        ax[0].set_title("Direkter Gehaltsvergleich")
        ax[0].set_xlabel("Messung")
        ax[0].set_ylabel("Gehalt")
        ax[0].grid(True, linestyle=':', alpha=0.5, zorder=1)
        ax[0].legend()

        # Plot 2: Scatterplot
        sns.scatterplot(x=df_aktuell["ZMB"], y=df_aktuell["INF"], ax=ax[1])
        ax[1].plot(df_aktuell["ZMB"], regression_line_ZMB, c="gray", 
                   label=f"r = {ZMB_INF_pearson}\nlinCCC = {linCCC(df_aktuell['ZMB'], df_aktuell['INF']):.3f}")
        ax[1].set_title("Regressionsanalyse")
        ax[1].set_xlabel("ZMB")
        ax[1].set_ylabel("INF")
        ax[1].legend(loc='lower right')
        ax[1].grid(True, linestyle=':', alpha=0.5, zorder=1)

        # Plot 3: Bland-Altman
        ax[2].scatter(mittelwerte, diff, color='black', alpha=0.6)
        ax[2].axhline(mean_diff, color='red', alpha=0.5, label=f'Bias: {mean_diff:.2f}')
        ax[2].axhline(loa_upper, color='gray', linestyle='--', label=f'+1.96 SD: {loa_upper:.2f}')
        ax[2].axhline(loa_lower, color='gray', linestyle='--', label=f'-1.96 SD: {loa_lower:.2f}')
        ax[2].set_title("Bland-Altman Plot")
        ax[2].set_ylabel("Differenz (ZMB - INF)")
        ax[2].set_xlabel("Mittelwert")
        ax[2].legend(loc='upper right')
        ax[2].grid(True, linestyle=':', alpha=0.5, zorder=1)

        fig.tight_layout()
        
        # Das hier ersetzt plt.show()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ein Fehler ist aufgetreten beim Lesen der Datei: {e}")

else:
    st.info("Bitte lade eine Excel-Datei hoch, um zu beginnen.")

