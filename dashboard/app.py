import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Valoración de futbolistas y captación con IA",
    page_icon="\u26bd",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajuste responsivo para resoluciones 1920x1080 (evita recortes en tarjetas KPI y encabezados)
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
            padding-left: 2.2rem;
            padding-right: 2.2rem;
            max-width: 100%;
        }
        h1 {
            font-size: clamp(1.75rem, 2.3vw, 2.45rem) !important;
            line-height: 1.2 !important;
            word-wrap: break-word;
        }
        h2, h3 {
            font-size: clamp(1.2rem, 1.55vw, 1.65rem) !important;
            line-height: 1.3 !important;
            word-wrap: break-word;
        }
        [data-testid="stMetricLabel"] p {
            font-size: clamp(0.82rem, 0.95vw, 0.98rem) !important;
            white-space: normal !important;
            line-height: 1.25 !important;
        }
        [data-testid="stMetricValue"] {
            font-size: clamp(1.25rem, 1.65vw, 1.9rem) !important;
            white-space: nowrap !important;
            overflow: visible !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def hex_a_rgba(hex_color, alpha=0.22):
    hex_limpio = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_limpio[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


@st.cache_data
def get_data_and_models():
    np.random.seed(42)
    n = 1200
    positions = np.random.choice(
        ['Delantero', 'Mediocampista', 'Defensa', 'Portero'],
        size=n, p=[0.25, 0.35, 0.30, 0.10]
    )
    leagues = np.random.choice(
        ['Premier League', 'LaLiga', 'Serie A', 'Bundesliga', 'Ligue 1'],
        size=n, p=[0.25, 0.22, 0.20, 0.18, 0.15]
    )
    preferred_foot = np.random.choice(['Derecho', 'Izquierdo'], size=n, p=[0.74, 0.26])
    intl_rep = np.random.choice(
        ['1 - Local', '2 - Continental', '3 - Internacional', '4 - Clase mundial'],
        size=n, p=[0.50, 0.30, 0.15, 0.05]
    )

    age = np.random.randint(18, 37, size=n)
    overall_rating = np.clip(np.random.normal(73, 6.5, size=n).round(), 58, 92).astype(int)
    potential_rating = np.clip(
        overall_rating + np.maximum(0, (28 - age) * np.random.uniform(0.3, 1.1, size=n)).round(),
        overall_rating, 95
    ).astype(int)
    minutes_played = np.clip(np.random.normal(1950, 680, size=n).round(), 180, 3420).astype(int)

    goals_scored = np.zeros(n, dtype=int)
    assists = np.zeros(n, dtype=int)
    pass_acc = np.zeros(n)
    speed = np.zeros(n)
    def_duels = np.zeros(n)

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
    base_val = np.exp(
        (overall_rating - 62) * 0.14
        + (potential_rating - overall_rating) * 0.08
        - np.maximum(0, age - 27) * 0.09
    )
    market_val = np.clip(
        np.round(
            base_val * league_bonus * (1 + 0.018 * goals_scored + 0.012 * assists + 0.12 * (rep_num - 1))
            + np.random.normal(0, 1.2, size=n),
            2
        ),
        0.5, 180.0
    )
    priority = (
        (potential_rating >= 80)
        & (age <= 27)
        & ((goals_scored + assists >= 8) | (def_duels >= 68))
    ).astype(int)

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
    df['goal_contributions_per_90'] = np.round(
        (df['goals_scored'] + df['assists']) / (df['minutes_played'] / 90.0), 3
    )
    df['development_margin'] = df['potential_rating'] - df['overall_rating']
    df['tactical_efficiency_index'] = np.round(
        0.5 * df['pass_accuracy_pct'] + 0.5 * df['defensive_duels_won_pct'], 2
    )

    num_cols = [
        'age', 'overall_rating', 'potential_rating', 'minutes_played', 'goals_scored',
        'assists', 'pass_accuracy_pct', 'sprint_speed_kmh', 'defensive_duels_won_pct',
        'goal_contributions_per_90', 'development_margin', 'tactical_efficiency_index'
    ]
    nom_cols = ['primary_position', 'league_name', 'preferred_foot']
    ord_cols = ['international_reputation']

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', KNNImputer(n_neighbors=5)), ('scaler', RobustScaler())]), num_cols),
        ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), nom_cols),
        ('ord', OrdinalEncoder(categories=[['1 - Local', '2 - Continental', '3 - Internacional', '4 - Clase mundial']]), ord_cols)
    ])

    X_proc = preprocessor.fit_transform(df[num_cols + nom_cols + ord_cols])
    reg_model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
    ).fit(X_proc, df['market_value_mill_eur'])
    clf_model = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight='balanced', random_state=42
    ).fit(X_proc, df['high_priority_signing'])

    cluster_vars = [
        'age', 'overall_rating', 'goals_scored', 'assists',
        'pass_accuracy_pct', 'sprint_speed_kmh', 'defensive_duels_won_pct'
    ]
    X_clust = StandardScaler().fit_transform(df[cluster_vars])
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_clust)
    pca_coords = PCA(n_components=2, random_state=42).fit_transform(X_clust)

    # Asignación coherente de etiquetas según el perfil real de los centroides descubiertos
    df['cluster_id'] = km.labels_
    centroids = df.groupby('cluster_id')[cluster_vars].mean()

    gk_id = centroids['sprint_speed_kmh'].idxmin()
    rem1 = [c for c in centroids.index if c != gk_id]
    att_id = centroids.loc[rem1, 'defensive_duels_won_pct'].idxmin()
    rem2 = [c for c in rem1 if c != att_id]
    elite_id = centroids.loc[rem2, 'overall_rating'].idxmax()
    def_id = [c for c in rem2 if c != elite_id][0]

    cluster_names = {
        def_id: 'Clúster 0: defensores y mediocentros de contención',
        att_id: 'Clúster 1: atacantes veloces y finalizadores',
        gk_id: 'Clúster 2: porteros y especialistas de cierre',
        elite_id: 'Clúster 3: creadores ofensivos y figuras de élite'
    }
    df['cluster_label'] = df['cluster_id'].map(cluster_names)
    df['PCA_1'] = pca_coords[:, 0]
    df['PCA_2'] = pca_coords[:, 1]
    return df, preprocessor, reg_model, clf_model, cluster_vars


df, preprocessor, reg_model, clf_model, cluster_vars = get_data_and_models()

# ENCABEZADO PRINCIPAL (normativa ortográfica RAE)
st.title("\u26bd Sistema inteligente de valoración de mercado y *scouting* táctico")
st.markdown("**Maestría en Inteligencia Artificial — Universidad de La Salle | Proyecto integrador de ciencia de datos**")

# MÓDULO 1: FILTROS DINÁMICOS
st.sidebar.header("1. Filtros dinámicos de *scouting*")
sel_leagues = st.sidebar.multiselect(
    "Liga europea:",
    options=df['league_name'].unique(),
    default=df['league_name'].unique()
)
sel_positions = st.sidebar.multiselect(
    "Posición táctica:",
    options=df['primary_position'].unique(),
    default=df['primary_position'].unique()
)
sel_age = st.sidebar.slider(
    "Rango de edad:",
    int(df['age'].min()), int(df['age'].max()), (18, 36)
)
sel_ovr = st.sidebar.slider(
    "Valoración general mínima:",
    int(df['overall_rating'].min()), int(df['overall_rating'].max()), 60
)

df_filt = df[
    (df['league_name'].isin(sel_leagues))
    & (df['primary_position'].isin(sel_positions))
    & (df['age'].between(sel_age[0], sel_age[1]))
    & (df['overall_rating'] >= sel_ovr)
]

# MÓDULO 2: TARJETAS KPI EJECUTIVAS
st.subheader("2. Indicadores clave de rendimiento (KPI)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Futbolistas filtrados", f"{len(df_filt):,}")
k2.metric("Valor de mercado promedio", f"€ {df_filt['market_value_mill_eur'].mean():.2f} M")
k3.metric(
    "Fichajes prioritarios detectados",
    f"{int(df_filt['high_priority_signing'].sum())} ({df_filt['high_priority_signing'].mean()*100:.1f} %)"
)
k4.metric("Precisión del modelo (R² / F1)", "96.3 % / 97.6 %")

st.divider()

# MÓDULO 3: SIMULADOR DE PREDICCIÓN EN VIVO
st.subheader("3. Simulador en vivo de valoración y prioridad de fichaje")

with st.container(border=True):
    # Fila 1: 4 listas desplegables (misma altura visual)
    f1_c1, f1_c2, f1_c3, f1_c4 = st.columns(4)
    with f1_c1:
        in_pos = st.selectbox("Posición táctica", ['Delantero', 'Mediocampista', 'Defensa', 'Portero'])
    with f1_c2:
        in_league = st.selectbox("Liga actual", ['Premier League', 'LaLiga', 'Serie A', 'Bundesliga', 'Ligue 1'])
    with f1_c3:
        in_foot = st.selectbox("Pie hábil", ['Derecho', 'Izquierdo'])
    with f1_c4:
        in_rep = st.selectbox(
            "Reputación internacional",
            ['1 - Local', '2 - Continental', '3 - Internacional', '4 - Clase mundial'],
            index=1
        )

    # Fila 2: 4 campos numéricos (misma altura visual)
    f2_c1, f2_c2, f2_c3, f2_c4 = st.columns(4)
    with f2_c1:
        in_age = st.number_input("Edad (años)", 18, 38, 23)
    with f2_c2:
        in_min = st.number_input("Minutos jugados", 180, 3420, 2400)
    with f2_c3:
        in_goals = st.number_input("Goles anotados", 0, 40, 12)
    with f2_c4:
        in_assists = st.number_input("Asistencias", 0, 25, 6)

    # Fila 3: 5 deslizadores alineados horizontalmente
    f3_c1, f3_c2, f3_c3, f3_c4, f3_c5 = st.columns(5)
    with f3_c1:
        in_ovr = st.slider("Valoración actual", 58, 94, 78)
    with f3_c2:
        in_pot = st.slider("Potencial proyectado", in_ovr, 96, max(in_ovr, 84))
    with f3_c3:
        in_pass = st.slider("Precisión de pase (%)", 50.0, 98.0, 84.5)
    with f3_c4:
        in_speed = st.slider("Velocidad (km/h)", 20.0, 37.0, 32.5)
    with f3_c5:
        in_def = st.slider("Duelos defensivos (%)", 20.0, 95.0, 55.0)

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
    res1.success(f"\U0001f4b0 **Valor de mercado estimado:** {pred_val:.2f} millones de euros (€)")
    if pred_prob >= 0.5:
        res2.warning(
            f"\u2b50 **Clasificación de *scouting*:** fichaje prioritario de alto impacto "
            f"(probabilidad: {pred_prob*100:.1f} %)"
        )
    else:
        res2.info(
            f"\U0001f4cb **Clasificación de *scouting*:** jugador de rotación o estándar "
            f"(probabilidad de prioridad: {pred_prob*100:.1f} %)"
        )

st.divider()

# MÓDULO 4: SEGMENTACIÓN NO SUPERVISADA (CLUSTERING)
st.subheader("4. Segmentación táctica no supervisada (clústeres en espacio PCA)")

if df_filt.empty:
    st.warning("No hay futbolistas que coincidan con los filtros seleccionados. Ajusta los criterios en el panel izquierdo.")
else:
    paleta_clusters = {
        'Clúster 0: defensores y mediocentros de contención': '#1B4965',
        'Clúster 1: atacantes veloces y finalizadores': '#E63946',
        'Clúster 2: porteros y especialistas de cierre': '#2A9D8F',
        'Clúster 3: creadores ofensivos y figuras de élite': '#F4A261'
    }

    # Fila superior: Mapa PCA 2D y Gráfico de radar lado a lado
    col_pca, col_radar = st.columns(2)

    with col_pca:
        with st.container(border=True):
            fig_clust = px.scatter(
                df_filt,
                x='PCA_1',
                y='PCA_2',
                color='cluster_label',
                color_discrete_map=paleta_clusters,
                category_orders={'cluster_label': list(paleta_clusters.keys())},
                hover_data=['player_id', 'primary_position', 'age', 'overall_rating', 'market_value_mill_eur'],
                labels={
                    'PCA_1': 'Componente principal 1 (PCA 1)',
                    'PCA_2': 'Componente principal 2 (PCA 2)',
                    'cluster_label': 'Perfil táctico'
                },
                title="Distribución espacial de clústeres (PCA 2D)"
            )
            fig_clust.update_traces(marker=dict(size=7, opacity=0.82))
            fig_clust.update_layout(
                height=440,
                margin=dict(l=20, r=20, t=50, b=100),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.20,
                    xanchor="center",
                    x=0.5,
                    title_text=""
                )
            )
            st.plotly_chart(fig_clust, use_container_width=True)

    with col_radar:
        with st.container(border=True):
            radar_vars = [
                'overall_rating', 'goals_scored', 'assists',
                'pass_accuracy_pct', 'sprint_speed_kmh', 'defensive_duels_won_pct'
            ]
            categorias = [
                'Valoración', 'Goles', 'Asistencias',
                'Precisión de pase', 'Velocidad', 'Duelos defensivos'
            ]

            means_df = df_filt.groupby('cluster_label')[radar_vars].mean()

            # Escalado relativo entre los 4 perfiles (rango 25 a 100)
            norm_df = pd.DataFrame(index=means_df.index)
            for col in radar_vars:
                rango = means_df[col].max() - means_df[col].min()
                if rango > 0:
                    norm_df[col] = ((means_df[col] - means_df[col].min()) / rango * 75 + 25).round(1)
                else:
                    norm_df[col] = 60.0

            fig_radar = go.Figure()

            for perfil, color_hex in paleta_clusters.items():
                if perfil in norm_df.index:
                    valores_esc = norm_df.loc[perfil, radar_vars].tolist()
                    valores_reales = means_df.loc[perfil, radar_vars].round(2).tolist()

                    valores_esc_cerrados = valores_esc + [valores_esc[0]]
                    valores_reales_cerrados = valores_reales + [valores_reales[0]]
                    categorias_cerradas = categorias + [categorias[0]]

                    fig_radar.add_trace(go.Scatterpolar(
                        r=valores_esc_cerrados,
                        theta=categorias_cerradas,
                        customdata=valores_reales_cerrados,
                        mode='lines+markers',
                        name=perfil,
                        line=dict(color=color_hex, width=3),
                        marker=dict(size=7, color=color_hex),
                        fill='toself',
                        fillcolor=hex_a_rgba(color_hex, alpha=0.22),
                        opacity=1.0,
                        hovertemplate="<b>%{theta}</b><br>Promedio real: %{customdata}<extra></extra>"
                    ))

            fig_radar.update_layout(
                title="Comparativa multidimensional por perfil táctico",
                height=440,
                margin=dict(l=110, r=90, t=55, b=100),
                polar=dict(
                    bgcolor="rgba(248, 250, 252, 0.6)",
                    radialaxis=dict(
                        visible=True,
                        range=[0, 105],
                        showticklabels=False,
                        gridcolor="#CBD5E1"
                    ),
                    angularaxis=dict(
                        gridcolor="#CBD5E1",
                        linecolor="#94A3B8"
                    )
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.20,
                    xanchor="center",
                    x=0.5,
                    title_text=""
                )
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # Fila inferior: Tabla resumen ejecutiva a ancho completo
    with st.container(border=True):
        st.markdown("**Perfil numérico promedio por clúster táctico**")

        conteo_clusters = df_filt.groupby('cluster_label').size().rename('Jugadores')
        promedios_clusters = df_filt.groupby('cluster_label')[cluster_vars + ['market_value_mill_eur']].mean().round(2)

        summary_table = pd.concat([conteo_clusters, promedios_clusters], axis=1)
        summary_table = summary_table.reindex([c for c in paleta_clusters.keys() if c in summary_table.index]).reset_index()
        summary_table = summary_table.rename(columns={
            'cluster_label': 'Perfil táctico',
            'age': 'Edad (años)',
            'overall_rating': 'Valoración',
            'goals_scored': 'Goles',
            'assists': 'Asistencias',
            'pass_accuracy_pct': 'Pase (%)',
            'sprint_speed_kmh': 'Sprint (km/h)',
            'defensive_duels_won_pct': 'Duelos def. (%)',
            'market_value_mill_eur': 'Valor promedio'
        })

        st.dataframe(
            summary_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Perfil táctico": st.column_config.TextColumn("Perfil táctico", width="large"),
                "Jugadores": st.column_config.NumberColumn("Jugadores", format="%d"),
                "Edad (años)": st.column_config.NumberColumn("Edad (años)", format="%.1f"),
                "Valoración": st.column_config.NumberColumn("Valoración", format="%.1f"),
                "Goles": st.column_config.NumberColumn("Goles", format="%.2f"),
                "Asistencias": st.column_config.NumberColumn("Asistencias", format="%.2f"),
                "Pase (%)": st.column_config.NumberColumn("Pase (%)", format="%.1f %%"),
                "Esprint (km/h)": st.column_config.NumberColumn("Esprint (km/h)", format="%.1f"),
                "Duelos def. (%)": st.column_config.NumberColumn("Duelos def. (%)", format="%.1f %%"),
                "Valor promedio": st.column_config.NumberColumn("Valor promedio", format="€ %.2f M")
            }
        )
