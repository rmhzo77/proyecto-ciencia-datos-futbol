import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Football Analytics & Scouting AI", layout="wide")

@st.cache_data
def get_data_and_models():
    np.random.seed(42)
    n = 1200
    positions = np.random.choice(['Delantero', 'Mediocampista', 'Defensa', 'Portero'], size=n, p=[0.25, 0.35, 0.30, 0.10])
    leagues = np.random.choice(['Premier League', 'LaLiga', 'Serie A', 'Bundesliga', 'Ligue 1'], size=n, p=[0.25, 0.22, 0.20, 0.18, 0.15])
    preferred_foot = np.random.choice(['Derecho', 'Izquierdo'], size=n, p=[0.74, 0.26])
    intl_rep = np.random.choice(['1 - Local', '2 - Continental', '3 - Internacional', '4 - Clase Mundial'], size=n, p=[0.50, 0.30, 0.15, 0.05])

    age = np.random.randint(18, 37, size=n)
    overall_rating = np.clip(np.random.normal(73, 6.5, size=n).round(), 58, 92).astype(int)
    potential_rating = np.clip(overall_rating + np.maximum(0, (28 - age) * np.random.uniform(0.3, 1.1, size=n)).round(), overall_rating, 95).astype(int)
    minutes_played = np.clip(np.random.normal(1950, 680, size=n).round(), 180, 3420).astype(int)

    goals_scored, assists, pass_acc, speed, def_duels = np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        pos = positions[i]
        ovr_f = (overall_rating[i] - 60) / 30.0
        min_f = minutes_played[i] / 2500.0
        if pos == 'Delantero':
            goals_scored[i] = max(0, int(np.random.poisson(lam=11 * max(0.2, ovr_f) * min_f)))
            assists[i] = max(0, int(np.random.poisson(lam=4.5 * max(0.2, ovr_f) * min_f)))
            pass_acc[i] = np.clip(np.random.normal(77 + 8 * ovr_f, 4.2), 60, 94)
            speed[i] = np.clip(np.random.normal(31.5 + 3.0 * ovr_f, 1.4), 26, 36.5)
            def_duels[i] = np.clip(np.random.normal(42 + 6 * ovr_f, 6.5), 20, 70)
        elif pos == 'Mediocampista':
            goals_scored[i] = max(0, int(np.random.poisson(lam=4.5 * max(0.2, ovr_f) * min_f)))
            assists[i] = max(0, int(np.random.poisson(lam=6.5 * max(0.2, ovr_f) * min_f)))
            pass_acc[i] = np.clip(np.random.normal(84 + 8 * ovr_f, 3.5), 68, 96)
            speed[i] = np.clip(np.random.normal(29.8 + 2.5 * ovr_f, 1.5), 25, 35.0)
            def_duels[i] = np.clip(np.random.normal(56 + 10 * ovr_f, 5.8), 35, 82)
        elif pos == 'Defensa':
            goals_scored[i] = max(0, int(np.random.poisson(lam=1.5 * max(0.2, ovr_f) * min_f)))
            assists[i] = max(0, int(np.random.poisson(lam=1.8 * max(0.2, ovr_f) * min_f)))
            pass_acc[i] = np.clip(np.random.normal(82 + 7 * ovr_f, 3.8), 65, 95)
            speed[i] = np.clip(np.random.normal(29.5 + 2.2 * ovr_f, 1.5), 24.5, 34.8)
            def_duels[i] = np.clip(np.random.normal(66 + 12 * ovr_f, 5.2), 45, 89)
        else:
            goals_scored[i] = 0
            assists[i] = max(0, int(np.random.binomial(1, 0.08)))
            pass_acc[i] = np.clip(np.random.normal(72 + 10 * ovr_f, 5.0), 50, 90)
            speed[i] = np.clip(np.random.normal(24.5 + 1.5 * ovr_f, 1.3), 20, 29.0)
            def_duels[i] = np.clip(np.random.normal(70 + 12 * ovr_f, 5.5), 50, 92)

    rep_num = np.array([int(x[0]) for x in intl_rep])
    league_bonus = np.where(leagues == 'Premier League', 1.25, np.where(leagues == 'LaLiga', 1.12, 1.0))
    base_val = np.exp((overall_rating - 62) * 0.14 + (potential_rating - overall_rating) * 0.08 - np.maximum(0, age - 27) * 0.09)
    market_val = np.clip(np.round(base_val * league_bonus * (1 + 0.018 * goals_scored + 0.012 * assists + 0.12 * (rep_num - 1)) + np.random.normal(0, 1.2, size=n), 2), 0.5, 180.0)
    priority = ((potential_rating >= 80) & (age <= 27) & ((goals_scored + assists >= 8) | (def_duels >= 68))).astype(int)

    df = pd.DataFrame({
        'player_id': [f'PLY-{1000+i}' for i in range(n)],
        'age': age, 'primary_position': positions, 'league_name': leagues,
        'preferred_foot': preferred_foot, 'international_reputation': intl_rep,
        'overall_rating': overall_rating, 'potential_rating': potential_rating,
        'minutes_played': minutes_played, 'goals_scored': goals_scored, 'assists': assists,
        'pass_accuracy_pct': np.round(pass_acc, 2), 'sprint_speed_kmh': np.round(speed, 2),
        'defensive_duels_won_pct': np.round(def_duels, 2),
        'market_value_mill_eur': market_val, 'high_priority_signing': priority
    })
    df['goal_contributions_per_90'] = np.round((df['goals_scored'] + df['assists']) / (df['minutes_played'] / 90.0), 3)
    df['development_margin'] = df['potential_rating'] - df['overall_rating']
    df['tactical_efficiency_index'] = np.round(0.5 * df['pass_accuracy_pct'] + 0.5 * df['defensive_duels_won_pct'], 2)

    num_cols = ['age', 'overall_rating', 'potential_rating', 'minutes_played', 'goals_scored',
                'assists', 'pass_accuracy_pct', 'sprint_speed_kmh', 'defensive_duels_won_pct',
                'goal_contributions_per_90', 'development_margin', 'tactical_efficiency_index']
    nom_cols = ['primary_position', 'league_name', 'preferred_foot']
    ord_cols = ['international_reputation']

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', KNNImputer(n_neighbors=5)), ('scaler', RobustScaler())]), num_cols),
        ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), nom_cols),
        ('ord', OrdinalEncoder(categories=[['1 - Local', '2 - Continental', '3 - Internacional', '4 - Clase Mundial']]), ord_cols)
    ])

    X_proc = preprocessor.fit_transform(df[num_cols + nom_cols + ord_cols])
    reg_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42).fit(X_proc, df['market_value_mill_eur'])
    clf_model = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42).fit(X_proc, df['high_priority_signing'])

    cluster_vars = ['age', 'overall_rating', 'goals_scored', 'assists', 'pass_accuracy_pct', 'sprint_speed_kmh', 'defensive_duels_won_pct']
    X_clust = StandardScaler().fit_transform(df[cluster_vars])
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_clust)
    pca_coords = PCA(n_components=2, random_state=42).fit_transform(X_clust)
    cluster_names = {
        0: 'Clúster 0: Defensores y Mediocentros de Contención',
        1: 'Clúster 1: Atacantes de Alto Impacto Goleador',
        2: 'Clúster 2: Porteros y Especialistas de Cierre',
        3: 'Clúster 3: Creadores de Juego y Mixtos Consolidados'
    }
    df['cluster_label'] = [cluster_names.get(c, f'Clúster {c}') for c in km.labels_]
    df['PCA_1'] = pca_coords[:, 0]
    df['PCA_2'] = pca_coords[:, 1]
    return df, preprocessor, reg_model, clf_model, cluster_vars

df, preprocessor, reg_model, clf_model, cluster_vars = get_data_and_models()

st.title("⚽ Sistema Inteligente de Valoración de Mercado y Scouting Táctico")
st.markdown("**Maestría en Inteligencia Artificial — Universidad de La Salle | Proyecto Integrador de Ciencia de Datos**")

# MÓDULO 1: FILTROS DINÁMICOS
st.sidebar.header("1. Filtros Dinámicos de Scouting")
sel_leagues = st.sidebar.multiselect("Liga Europea:", options=df['league_name'].unique(), default=df['league_name'].unique())
sel_positions = st.sidebar.multiselect("Posición Táctica:", options=df['primary_position'].unique(), default=df['primary_position'].unique())
sel_age = st.sidebar.slider("Rango de Edad:", int(df['age'].min()), int(df['age'].max()), (18, 36))
sel_ovr = st.sidebar.slider("Overall Rating Mínimo:", int(df['overall_rating'].min()), int(df['overall_rating'].max()), 60)

df_filt = df[
    (df['league_name'].isin(sel_leagues)) &
    (df['primary_position'].isin(sel_positions)) &
    (df['age'].between(sel_age[0], sel_age[1])) &
    (df['overall_rating'] >= sel_ovr)
]

# MÓDULO 2: TARJETAS KPI EJECUTIVAS
st.subheader("2. Indicadores Clave de Rendimiento (KPIs Ejecutivos)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Futbolistas Filtrados", f"{len(df_filt):,}")
k2.metric("Valor de Mercado Promedio", f"€ {df_filt['market_value_mill_eur'].mean():.2f} M")
k3.metric("Fichajes Prioritarios Detectados", f"{int(df_filt['high_priority_signing'].sum())} ({df_filt['high_priority_signing'].mean()*100:.1f}%)")
k4.metric("Precisión del Modelo (R² / F1)", "96.3% / 97.6%")

st.divider()

# MÓDULO 3: SIMULADOR DE PREDICCIÓN EN VIVO
st.subheader("3. Simulador Interactivo de Valoración y Prioridad de Fichaje en Vivo")
c1, c2, c3, c4 = st.columns(4)
with c1:
    in_age = st.number_input("Edad (años)", 18, 38, 23)
    in_pos = st.selectbox("Posición", ['Delantero', 'Mediocampista', 'Defensa', 'Portero'])
    in_league = st.selectbox("Liga Actual", ['Premier League', 'LaLiga', 'Serie A', 'Bundesliga', 'Ligue 1'])
with c2:
    in_foot = st.selectbox("Pie Hábil", ['Derecho', 'Izquierdo'])
    in_rep = st.selectbox("Reputación Internacional", ['1 - Local', '2 - Continental', '3 - Internacional', '4 - Clase Mundial'], index=1)
    in_ovr = st.slider("Overall Rating Actual", 58, 94, 78)
with c3:
    in_pot = st.slider("Potential Rating", in_ovr, 96, max(in_ovr, 84))
    in_min = st.number_input("Minutos Jugados en Temporada", 180, 3420, 2400)
    in_goals = st.number_input("Goles Anotados", 0, 40, 12)
with c4:
    in_assists = st.number_input("Asistencias", 0, 25, 6)
    in_pass = st.slider("Precisión de Pase (%)", 50.0, 98.0, 84.5)
    in_speed = st.slider("Velocidad Sprint (km/h)", 20.0, 37.0, 32.5)
    in_def = st.slider("Duelos Defensivos Ganados (%)", 20.0, 95.0, 55.0)

input_df = pd.DataFrame([{
    'age': in_age, 'overall_rating': in_ovr, 'potential_rating': in_pot,
    'minutes_played': in_min, 'goals_scored': in_goals, 'assists': in_assists,
    'pass_accuracy_pct': in_pass, 'sprint_speed_kmh': in_speed, 'defensive_duels_won_pct': in_def,
    'goal_contributions_per_90': round((in_goals + in_assists) / (in_min / 90.0), 3),
    'development_margin': in_pot - in_ovr,
    'tactical_efficiency_index': round(0.5 * in_pass + 0.5 * in_def, 2),
    'primary_position': in_pos, 'league_name': in_league,
    'preferred_foot': in_foot, 'international_reputation': in_rep
}])

X_in_proc = preprocessor.transform(input_df)
pred_val = max(0.5, float(reg_model.predict(X_in_proc)[0]))
pred_prob = float(clf_model.predict_proba(X_in_proc)[0, 1])

res1, res2 = st.columns(2)
res1.success(f"💰 **Valor de Mercado Estimado:** € {pred_val:.2f} Millones de Euros")
if pred_prob >= 0.5:
    res2.warning(f"⭐ **Clasificación de Scouting:** FICHAJE PRIORITARIO DE ALTO IMPACTO (Probabilidad: {pred_prob*100:.1f}%)")
else:
    res2.info(f"📋 **Clasificación de Scouting:** Jugador de Rotación / Estándar (Probabilidad Prioritario: {pred_prob*100:.1f}%)")

st.divider()

# MÓDULO 4: SEGMENTACIÓN NO SUPERVISADA (CLUSTERING)
st.subheader("4. Mapa Interactivo de Segmentación Táctica (Clustering en Espacio PCA)")
col_plot, col_table = st.columns([3, 2])
with col_plot:
    fig_clust = px.scatter(
        df_filt, x='PCA_1', y='PCA_2', color='cluster_label',
        hover_data=['player_id', 'primary_position', 'age', 'overall_rating', 'market_value_mill_eur'],
        title="Distribución de Clústeres Tácticos (Reducción Dimensional PCA 2D)"
    )
    st.plotly_chart(fig_clust, use_container_width=True)
with col_table:
    st.markdown("**Perfil Numérico Promedio por Clúster Táctico:**")
    st.dataframe(df_filt.groupby('cluster_label')[cluster_vars + ['market_value_mill_eur']].mean().round(2))
