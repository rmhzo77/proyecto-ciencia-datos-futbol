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
    page_icon="⚽",
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
st.title("⚽
