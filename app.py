import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

st.set_page_config(page_title="ZMB Puffer Validierung", layout="centered")
st.title("Validierung: ZMB vs. INF3 Puffer")
st.markdown("Lade die Validierungsdaten hoch, um die Analyse für **Chargen** oder **PK** zu starten.")

# Datei Upload & Input
with st.sidebar:
    st.header("Einstellungen")
    uploaded_file = st.file_uploader("Excel Datei hochladen", type="xlsx")
    st.divider()
    auswahl = st.radio("Welches Set analysieren?", ["Chargen", "PK", "Beides"])

# Daten laden
if uploaded_file is not None:
    try:
       df_daten = pd.read_excel(uploaded_file, sheet_name="Daten")
        
        if datenset == "Chargen":
            df_sub = df_daten[df_daten["Probe"].isin(["Gardasil 9", "Gardasil"])].copy()
        elif datenset == "PK":
            df_sub = df_daten[df_daten["Probe"] == "Positivkontrolle"].copy()
        elif datenset == "Beides":
            df_sub = df_daten.copy()
        
        if "Bemerkung" in df_sub.columns:
            df_sub = df_sub[df_sub["Bemerkung"].isna()]
            
        cols_to_clean = ["Gehalt (U/ml)"]
        
        if "Gehalt (U/ml)" not in df_sub.columns or "Pufferansatz" not in df_sub.columns:
            st.error("Fehler: Die Spalten 'Gehalt (U/ml)' oder 'Pufferansatz' fehlen in der Datei.")
            st.stop()

        df_sub[cols_to_clean] = df_sub[cols_to_clean].apply(pd.to_numeric, errors='coerce')
        df_sub = df_sub.dropna(subset=cols_to_clean)

        df_zmb = df_sub[df_sub["Pufferansatz"] == "ZMB"][["Charge", "Probe", "Gehalt (U/ml)"]]
        df_inf3 = df_sub[df_sub["Pufferansatz"] == "INF3"][["Charge", "Probe", "Gehalt (U/ml)"]]

        if df_zmb.empty or df_inf3.empty:
            st.warning("Keine passenden Daten für ZMB oder INF3 gefunden. Bitte Filter prüfen.")
            st.stop()

        df_sub = df_sub.sort_values(by=["Typ"]) if "Typ" in df_sub.columns else df_sub

        df_zmb['Rep_ID'] = df_zmb.groupby(['Charge', 'Probe']).cumcount()
        df_inf3['Rep_ID'] = df_inf3.groupby(['Charge', 'Probe']).cumcount()

        df_zmb = df_zmb.rename(columns={"Gehalt (U/ml)": "ZMB"})
        df_inf3 = df_inf3.rename(columns={"Gehalt (U/ml)": "INF3"})

        df_merged = pd.merge(df_zmb, df_inf3, on=["Charge", "Probe", "Rep_ID"], how="inner")
        df_merged = df_merged.drop(columns=["Rep_ID"])

        cols_final = ["ZMB", "INF3"]
        df_merged[cols_final] = df_merged[cols_final].apply(pd.to_numeric, errors='coerce')
        df_merged = df_merged.dropna(subset=cols_final)

        ZMB = df_merged["ZMB"].to_numpy()
        INF3 = df_merged["INF3"].to_numpy()
        Chargen_Labels = df_merged["Charge"].to_numpy()
        n = len(ZMB)

        # Log normalisierung für Äquivalenzgrenzen
        log_ZMB, log_INF3 = np.log(ZMB), np.log(INF3)
        log_diff = log_ZMB - log_INF3

        mean_log_diff = np.mean(log_diff)
        se_log_diff = stats.sem(log_diff)
        ci_log_low, ci_log_high = stats.t.interval(0.95, n-1, loc=mean_log_diff, scale=se_log_diff)

        point_estimate_ratio = np.exp(mean_log_diff)
        ci_ratio_low = np.exp(ci_log_low)
        ci_ratio_high = np.exp(ci_log_high)

        # Prozentuale Differenz (für Bland-Altman)
        diff_prozent = (ZMB - INF3) / INF3 * 100
        mean_diff_pct = np.mean(diff_prozent)
        std_diff_pct = np.std(diff_prozent, ddof=1)

        # Lin CCC Funktion
        def Lin_CCC(x, y):
            x_mean, y_mean = np.mean(x), np.mean(y)
            x_var, y_var = np.var(x, ddof=1), np.var(y, ddof=1)
            cov_xy = np.cov(x, y)[0, 1]
            ccc = (2 * cov_xy) / (x_var + y_var + (x_mean - y_mean)**2)
            return np.round(ccc, 4)

        ccc_val = Lin_CCC(ZMB, INF3)
        pearson_r = np.round(np.corrcoef(ZMB, INF3)[0, 1], 4)

        # Regression Line
        m, b = np.polyfit(ZMB, INF3, 1)
        regression_line_ZMB = m * ZMB + b

        #  Statistische Auswertung
        st.divider()
        st.subheader("Statistische Auswertung")

        c1, c2, c3 = st.columns(3)
        c1.metric("Anzahl Messungen (n)", n)
        c2.metric("Lin's CCC", f"{ccc_val}")
        c3.metric("Ratio (ZMB/INF3)", f"{point_estimate_ratio:.4f}", help="Geometrisches Mittel der Ratios")

        st.caption(f"95% CI der Ratio: [{ci_ratio_low:.4f} bis {ci_ratio_high:.4f}]")

        # Äquivalenz Check
        ist_aequivalent = (ci_ratio_low >= 0.80) and (ci_ratio_high <= 1.25)
        
        st.write("---")
        st.write("**Prüfung Äquivalenzgrenzen (80% - 125%)**")
        if ist_aequivalent:
            st.success(f"✅ ÄQUIVALENZ BESTÄTIGT: Das 95% CI ({ci_ratio_low*100:.1f}% - {ci_ratio_high*100:.1f}%) liegt vollständig innerhalb von 80-125%.")
        else:
            st.error(f"❌ KEINE ÄQUIVALENZ: Das 95% CI ({ci_ratio_low*100:.1f}% - {ci_ratio_high*100:.1f}%) liegt teilweise außerhalb von 80-125%.")

        # Plots
        st.subheader("Grafische Auswertung")

        # Plot Setup
        Messungen = np.arange(len(ZMB))
        x_bar = np.arange(len(Messungen))
        breite = 0.35
        position_ZMB = x_bar - breite/2
        position_INF3 = x_bar + breite/2
        
        # Figure Setup
        fig, ax = plt.subplots(3, 1, figsize=(6, 14))
        
        # Plot 1: Barplot
        ax[0].bar(position_ZMB, ZMB], breite, color="#297fc1", zorder=3, label="ZMB")
        ax[0].bar(position_INF3, INF3, breite, color="#ffb64e", zorder=3, label="INF3")
        ax[0].set_title("Direkter Gehaltsvergleich")
        ax[0].set_xlabel("Messung")
        ax[0].set_ylabel("Gehalt (U/ml")
        if datenset == "Beides": ax[0].set_yscale("log", base=10), ax[0].set_ylim(10, 2000)
        ax[0].set_xticks(x_bar)
        ax[0].set_xticklabels(Chargen_Labels, rotation=45, ha='right', fontsize=6)  
        ax[0].legend(loc="lower right", fontsize="small")
        ax[0].grid(axis="y", ls=":", which="major", lw=1, alpha=0.5, zorder=1)

        # Plot 2: Scatterplot
        ax[1].scatter(ZMB, INF3)
        ax[1].set_title("Regressionsanalyse")
        ax[1].set_xlabel("ZMB", fontsize=10)
        ax[1].set_ylabel("INF3", fontsize=10)
        ax[1].plot(ZMB, regression_line_ZMB, c="gray", label=f"r = {pearson_r} ; linCCC = {ccc_val}")
        ax[1].legend(loc='lower right')
        ax[1].grid(True, linestyle=':', alpha=0.5, zorder=1)

        # Plot 3: Bland-Altman
        loa_upper = mean_diff_pct + 1.96 * std_diff_pct
        loa_lower = mean_diff_pct - 1.96 * std_diff_pct
        
        ax[2].scatter(INF3, diff_prozent, color='black', alpha=0.6)
        ax[2].axhline(mean_diff_pct, color='red', alpha=0.5, linestyle="-", lw=1.5 label=f'Bias: {mean_diff_pct:.2f}')
        ax[2].axhline(loa_upper, color='gray', linestyle='--', label=f'+1.96 SD: {loa_upper:.2f}')
        ax[2].axhline(loa_lower, color='gray', linestyle='--', label=f'-1.96 SD: {loa_lower:.2f}')
        ax[2].axhspan(-20, 25, color='green', alpha=0.05, label='Äquivalenzzone (80-125%)')
        ax[2].set_title("Prozentualer Bland-Altman Plot", fontsize=11)
        ax[2].set_ylabel("Abweichung (%)")
        ax[2].set_xlabel("Referenzgehalt (INF3)")
        ax[2].legend(loc='upper right', bbox_to_anchor=(1, 0.95), fontsize=8, framealpha=0.9)
        ax[2].grid(True, linestyle=':', alpha=0.5, zorder=1)

        # Plots anpassungen
        for a in ax:
            a.tick_params(axis="both", labelsize=9)
        
        fig.tight_layout()
        
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ein Fehler ist aufgetreten beim Lesen der Datei: {e}")

else:
    st.info("Bitte lade eine Excel-Datei hoch, um zu beginnen.")


