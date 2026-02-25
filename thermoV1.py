import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy.optimize import brentq
from scipy.signal import argrelextrema

# ==========================================
# CONFIGURATION STREAMLIT
# ==========================================
st.set_page_config(page_title="Modélisation des Gaz Réels", layout="wide")

R = 0.08314

# ==========================================
# 1. MOTEUR PHYSIQUE
# ==========================================
def P_vdw(V, T, a, b):
    return (R * T) / (V - b) - a / (V**2)

def P_ideal(V, T):
    return (R * T) / V

def primitive_VdW(V, T, a, b):
    return R * T * np.log(np.abs(V - b)) + a / V

def difference_aires(P_test, T, a, b):
    coeffs = [P_test, -(P_test * b + R * T), a, -a * b]
    roots = np.roots(coeffs)
    r_reelles = np.sort(roots[np.isreal(roots)].real)
    if len(r_reelles) != 3: 
        return 1e9
    aire_courbe = primitive_VdW(r_reelles[2], T, a, b) - primitive_VdW(r_reelles[0], T, a, b)
    aire_rect = P_test * (r_reelles[2] - r_reelles[0])
    return aire_courbe - aire_rect

def trouver_plateau(T, a, b):
    v_scan = np.linspace(b + 0.001, 1.5, 2000)
    p_scan = P_vdw(v_scan, T, a, b)
    
    idx_max = argrelextrema(p_scan, np.greater)[0]
    idx_min = argrelextrema(p_scan, np.less)[0]
    
    if len(idx_max) == 0 or len(idx_min) == 0:
        return None # Supercritique
        
    P_high = p_scan[idx_max[0]]
    P_low = max(p_scan[idx_min[0]], 0.01)
    
    try:
        Psat = brentq(difference_aires, P_low + 0.01, P_high - 0.01, args=(T, a, b))
        coeffs = [Psat, -(Psat * b + R * T), a, -a * b]
        roots_fin = np.sort(np.roots(coeffs)[np.isreal(np.roots(coeffs))].real)
        return Psat, roots_fin[0], roots_fin[1], roots_fin[2]
    except ValueError:
        return None

def coordonnees_critiques(a, b):
    """Calcule les paramètres critiques théoriques"""
    Tc = (8 * a) / (27 * R * b)
    Vc = 3 * b
    Pc = a / (27 * b**2)
    return Tc, Vc, Pc

# ==========================================
# 2. MENU LATÉRAL (SIDEBAR)
# ==========================================
st.sidebar.title("⚙️ Paramètres de l'Étude")

# Choix du mode d'affichage
mode = st.sidebar.radio("Choix du Mode", ["1️⃣ Isotherme Unique (Détaillé)", "2️⃣ Plage de Températures (Cloche)"])

st.sidebar.markdown("---")
st.sidebar.subheader("Carte d'identité du Gaz")
a = st.sidebar.number_input("Paramètre 'a' (Attraction)", value=3.640, step=0.1, format="%.3f")
b = st.sidebar.number_input("Paramètre 'b' (Covolume)", value=0.0427, step=0.001, format="%.4f")

# Calcul des coordonnées critiques
Tc_crit, Vc_crit, Pc_crit = coordonnees_critiques(a, b)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Point Critique Théorique")
st.sidebar.info(f"**Tc** : {Tc_crit:.2f} K\n\n**Pc** : {Pc_crit:.2f} bar\n\n**Vc** : {Vc_crit:.4f} L/mol")

# ==========================================
# 3. AFFICHAGE DYNAMIQUE
# ==========================================

# Limites fixes pour le graphe basées sur les constantes critiques pour éviter les sauts d'échelle
V_MAX_PLOT = Vc_crit * 6
P_MAX_PLOT = Pc_crit * 2.5

if mode == "1️⃣ Isotherme Unique (Détaillé)":
    st.title("🔍 Étude d'une Isotherme de Van der Waals")
    
    # Curseur basé sur Tc
    T = st.slider("Température de l'isotherme (K)", min_value=100.0, max_value=800.0, value=280.0, step=1.0)
    
    resultat = trouver_plateau(T, a, b)
    
    # Préparation de la figure (toujours affichée)
    fig = plt.figure(figsize=(12, 7), facecolor='white')
    V = np.linspace(b*1.05, V_MAX_PLOT, 2000)
    
    plt.plot(V, P_ideal(V, T), 'k--', alpha=0.3, label="Gaz Parfait")
    plt.plot(V, P_vdw(V, T, a, b), 'b-', alpha=0.6, label="Van der Waals")

    if resultat:
        P_sat, V_liq, V_mid, V_gaz = resultat
        
        # Affichage des résultats
        col1, col2, col3 = st.columns(3)
        col1.metric("Pression de Saturation", f"{P_sat:.2f} bar")
        col2.metric("Volume Liquide", f"{V_liq:.4f} L/mol")
        col3.metric("Volume Gaz", f"{V_gaz:.4f} L/mol")
        
        st.markdown("---")
        
        # Tracé de la construction de Maxwell
        plt.plot([V_liq, V_gaz], [P_sat, P_sat], 'r-', linewidth=2.5, label=f"Plateau ($P_{{sat}}={P_sat:.2f}$ bar)")
        
        # Remplissage des aires
        v_creux = np.linspace(V_mid, V_gaz, 200)
        plt.fill_between(v_creux, P_vdw(v_creux, T, a, b), P_sat, color='green', alpha=0.3, label="Aires égales")
        v_bosse = np.linspace(V_liq, V_mid, 200)
        plt.fill_between(v_bosse, P_vdw(v_bosse, T, a, b), P_sat, color='orange', alpha=0.3)

        plt.scatter([V_liq, V_gaz], [P_sat, P_sat], color='black', zorder=10)
        
        # Décalage dynamique du texte basé sur Pc_crit pour éviter de sortir du cadre
        offset_y = Pc_crit * 0.05
        plt.text(V_liq, P_sat - offset_y, '$V_{liq}$', ha='center', color='red', fontsize=12)
        plt.text(V_gaz, P_sat - offset_y, '$V_{gaz}$', ha='center', color='red', fontsize=12)

    else:
        st.info("ℹ️ État Supercritique : Le fluide est dans une phase unique, il n'y a pas de transition d'état liquide/vapeur à cette température.")
        st.markdown("---")

    # Configuration des axes fixes
    plt.title(f"Isotherme à {T} K (a={a}, b={b})", fontsize=14, fontweight='bold')
    plt.xlabel("Volume Molaire (L/mol)")
    plt.ylabel("Pression (bar)")
    plt.xlim(0, V_MAX_PLOT)
    plt.ylim(0, P_MAX_PLOT)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    st.pyplot(fig)

elif mode == "2️⃣ Plage de Températures (Cloche)":
    st.title("📈 Diagramme de Phase Complet (Courbe Binodale)")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        T_min = st.number_input("Température Min (K)", value=250.0)
    with col_t2:
        T_max = st.number_input("Température Max (K)", value=float(int(Tc_crit)))
    with col_t3:
        pas_T = st.selectbox("Pas des isothermes (K)", options=[2, 5, 10, 20], index=1)

    # Calcul de la cloche
    temps_cloche = np.linspace(T_min, Tc_crit * 0.999, 40)
    
    # Nouvelles listes sécurisées
    t_valides_list = []
    v_liq_list, v_gaz_list, p_sat_list = [], [], []
    
    with st.spinner("Calcul des équilibres..."):
        for T_i in temps_cloche:
            res = trouver_plateau(T_i, a, b)
            if res:
                # On sauvegarde uniquement les températures qui ont convergé
                t_valides_list.append(T_i) 
                p_sat_list.append(res[0])
                v_liq_list.append(res[1])
                v_gaz_list.append(res[3])
    
    # Ajout du point critique pour fermer la binodale
    t_valides_list.append(Tc_crit)
    v_liq_list.append(Vc_crit)
    v_gaz_list.append(Vc_crit)
    p_sat_list.append(Pc_crit)

    fig2 = plt.figure(figsize=(12, 8), facecolor='white')
    V_plot = np.linspace(b + 0.005, V_MAX_PLOT, 1000)
    
    # 1. Isothermes (Bleu acier)
    T_reseau = np.arange(T_min, Tc_crit * 1.5, pas_T)
    for T_iso in T_reseau:
        plt.plot(V_plot, P_vdw(V_plot, T_iso, a, b), color='steelblue', alpha=0.3, lw=1.2)
    
    # 2. Plateaux (Crimson)
    for i in range(len(p_sat_list)-1):
        plt.plot([v_liq_list[i], v_gaz_list[i]], [p_sat_list[i], p_sat_list[i]], color='crimson', alpha=0.8, lw=1.5)

    # 3. Binodale (Cloche noire)
    v_binodale = v_liq_list + v_gaz_list[::-1]
    p_binodale = p_sat_list + p_sat_list[::-1]
    plt.plot(v_binodale, p_binodale, color='black', lw=2.5, linestyle='--', label="Courbe Binodale")

    # 4. Point Critique (Étoile dorée)
    plt.scatter(Vc_crit, Pc_crit, color='gold', s=300, marker='*', edgecolor='black', zorder=15, label='POINT CRITIQUE')
    
    offset_y = Pc_crit * 0.05
    plt.annotate(f"C ({Pc_crit:.1f} bar, {Vc_crit:.3f} L/mol)", 
                 xy=(Vc_crit, Pc_crit), xytext=(Vc_crit+0.05, Pc_crit + offset_y),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    # Configuration des axes fixes
    plt.xlim(0, V_MAX_PLOT)
    plt.ylim(0, P_MAX_PLOT)
    plt.xlabel("Volume molaire $V$ (L/mol)")
    plt.ylabel("Pression $P$ (bar)")
    plt.title(f"Diagramme (P,V) de {T_min} K à {T_max} K", fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig2)
    
    # Tableau de données
    st.subheader("📊 Données de saturation")
    df = pd.DataFrame({
        'T (K)': np.round(t_valides_list, 1),
        'P_sat (bar)': np.round(p_sat_list, 2),
        'V_liq': np.round(v_liq_list, 4),
        'V_gaz': np.round(v_gaz_list, 4)
    }).sort_values('T (K)', ascending=False)
    st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    pass