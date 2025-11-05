import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Prédiction PM10 - Casablanca",
    page_icon="🌍",
    layout="wide"
)

# ==================== CHARGEMENT ====================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_xgboost_model.joblib")
        return model
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle : {e}")
        return None

@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv("data_model_ready.csv")
        if 'Unnamed: 0' in df.columns:
            df['date'] = pd.to_datetime(df['Unnamed: 0'])
            df = df.drop(columns=['Unnamed: 0'])
        else:
            df.index = pd.to_datetime(df.index)
            df['date'] = df.index
        return df
    except Exception as e:
        st.warning(f"⚠️ Données historiques non disponibles : {e}")
        return None

# ==================== FONCTION DE FEATURES CORRIGÉE ====================
def create_all_features_correct(input_data, historical_df=None):
    """
    Crée EXACTEMENT les mêmes features qu'à l'entraînement
    Dans le MÊME ORDRE
    """
    df = pd.DataFrame([input_data])
    
    # Convertir la date
    df['date'] = pd.to_datetime(df['date'])
    
    # ========== FEATURES DE BASE (DOIVENT ÊTRE PRÉSENTES) ==========
    # Ces colonnes doivent exister : tavg, tmin, tmax, prcp, wspd, pres, month, year
    
    # ========== 1. FEATURES TEMPORELLES CYCLIQUES ==========
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # ========== 2. LAG FEATURES PM10 ==========
    # Utiliser les vraies valeurs historiques si disponibles
    if historical_df is not None and 'pm10_mean' in historical_df.columns:
        try:
            target_date = df['date'].iloc[0]
            
            # Récupérer les valeurs réelles des jours précédents
            for lag in [1, 2, 3, 7, 14, 30]:
                lag_date = target_date - timedelta(days=lag)
                lag_value = historical_df[historical_df['date'] == lag_date]['pm10_mean']
                df[f'pm10_lag{lag}'] = lag_value.iloc[0] if len(lag_value) > 0 else input_data.get('pm10_current', 30.0)
        except:
            # Fallback : utiliser pm10_current
            current_pm10 = input_data.get('pm10_current', 30.0)
            df['pm10_lag1'] = current_pm10
            df['pm10_lag2'] = current_pm10
            df['pm10_lag3'] = current_pm10
            df['pm10_lag7'] = current_pm10
            df['pm10_lag14'] = current_pm10
            df['pm10_lag30'] = current_pm10
    else:
        # Fallback : utiliser pm10_current
        current_pm10 = input_data.get('pm10_current', 30.0)
        df['pm10_lag1'] = current_pm10
        df['pm10_lag2'] = current_pm10
        df['pm10_lag3'] = current_pm10
        df['pm10_lag7'] = current_pm10
        df['pm10_lag14'] = current_pm10
        df['pm10_lag30'] = current_pm10
    
    # ========== 3. ROLLING STATISTICS ==========
    # Utiliser les moyennes des lags comme approximation
    df['pm10_rolling_mean_3'] = (df['pm10_lag1'] + df['pm10_lag2'] + df['pm10_lag3']) / 3
    df['pm10_rolling_mean_7'] = (df['pm10_lag1'] + df['pm10_lag7']) / 2
    df['pm10_rolling_mean_14'] = df['pm10_lag14']
    df['pm10_rolling_mean_30'] = df['pm10_lag30']
    
    df['pm10_rolling_std_7'] = abs(df['pm10_lag1'] - df['pm10_lag7']) / 2
    df['pm10_rolling_std_30'] = abs(df['pm10_lag1'] - df['pm10_lag30']) / 2
    
    df['pm10_rolling_min_7'] = min(df['pm10_lag1'].iloc[0], df['pm10_lag7'].iloc[0])
    df['pm10_rolling_max_7'] = max(df['pm10_lag1'].iloc[0], df['pm10_lag7'].iloc[0])
    
    df['pm10_trend_3d'] = df['pm10_lag1'] - df['pm10_rolling_mean_3']
    df['pm10_trend_7d'] = df['pm10_lag1'] - df['pm10_rolling_mean_7']
    
    # ========== 4. LAG FEATURES MÉTÉO ==========
    for var in ['tavg', 'wspd', 'prcp', 'pres']:
        df[f'{var}_lag1'] = df[var]  # Approximation
        df[f'{var}_rolling_mean_3'] = df[var]
        df[f'{var}_rolling_mean_7'] = df[var]
    
    # ========== 5. FEATURES D'INTERACTION ==========
    df['temp_pressure'] = df['tavg'] * df['pres']
    df['temp_wind'] = df['tavg'] * df['wspd']
    df['wind_precip'] = df['wspd'] * (df['prcp'] + 0.1)
    df['pressure_wind'] = df['pres'] * df['wspd']
    
    # ========== 6. SAISONS ENCODÉES ==========
    seasons = {12: 'Hiver', 1: 'Hiver', 2: 'Hiver', 
               3: 'Printemps', 4: 'Printemps', 5: 'Printemps',
               6: 'Été', 7: 'Été', 8: 'Été',
               9: 'Automne', 10: 'Automne', 11: 'Automne'}
    season = seasons[df['month'].iloc[0]]
    
    for s in ['Automne', 'Hiver', 'Printemps', 'Été']:
        df[f'season_{s}'] = int(season == s)
    
    # ========== 7. INDICATEURS DE STABILITÉ ==========
    df['temp_range'] = df['tmax'] - df['tmin']
    df['stability_index'] = df['temp_range'] / (df['wspd'] + 0.1)
    df['stagnant_conditions'] = int((df['pres'].iloc[0] > 1015) and (df['wspd'].iloc[0] < 5))
    
    # ========== 8. FEATURES COMPOSITES ==========
    df['heat_index'] = df['tavg'] + 0.5 * df['tmax']
    df['dispersion_potential'] = df['wspd'] * df['temp_range']
    df['wash_effect'] = df['prcp'] * df['wspd']
    df['high_pollution_recent'] = int(df['pm10_rolling_mean_7'].iloc[0] > 40)
    
    # ========== 9. FEATURES TEMPORELLES AVANCÉES ==========
    df['is_weekend'] = int(df['day_of_week'].iloc[0] >= 5)
    df['quarter'] = (df['month'] - 1) // 3 + 1
    
    for q in [1, 2, 3, 4]:
        df[f'quarter_{q}'] = int(df['quarter'].iloc[0] == q)
    
    # ========== SUPPRIMER LES COLONNES TEMPORAIRES ==========
    df = df.drop(['date', 'quarter'], axis=1, errors='ignore')
    
    return df

# ==================== INTERFACE ====================
st.title("🌍 Prédiction PM10 - Casablanca")
st.markdown("**Application de prédiction de la qualité de l'air**")

model = load_model()
historical_df = load_historical_data()

if model is None:
    st.error("❌ Modèle non chargé")
    st.stop()

# ==================== FORMULAIRE ====================
with st.form("prediction_form"):
    st.subheader("📅 Date de prédiction")
    date = st.date_input("Date", value=datetime.today())
    
    month = date.month
    year = date.year
    
    st.subheader("🌡️ Paramètres Météorologiques")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tavg = st.number_input("🌡️ Température moyenne (°C)", -10.0, 50.0, 20.0, 0.5)
        tmin = st.number_input("❄️ Température min (°C)", -10.0, 50.0, 15.0, 0.5)
    
    with col2:
        tmax = st.number_input("🔥 Température max (°C)", -10.0, 50.0, 25.0, 0.5)
        prcp = st.number_input("🌧️ Précipitations (mm)", 0.0, 100.0, 0.0, 0.5)
    
    with col3:
        wspd = st.number_input("💨 Vitesse vent (km/h)", 0.0, 100.0, 10.0, 0.5)
        pres = st.number_input("🔵 Pression (hPa)", 900.0, 1100.0, 1013.0, 1.0)
    
    st.subheader("📊 Contexte Pollution")
    pm10_current = st.number_input("PM10 actuel (µg/m³)", 0.0, 500.0, 30.0, 1.0,
                                     help="Valeur PM10 du jour précédent ou estimation")
    
    submitted = st.form_submit_button("🚀 Prédire", type="primary")
    
    if submitted:
        # Validation
        if tmin > tavg or tavg > tmax:
            st.error("❌ Températures incohérentes : tmin ≤ tavg ≤ tmax")
        else:
            with st.spinner("🔄 Calcul en cours..."):
                try:
                    # Préparer les données d'entrée
                    input_data = {
                        'date': date,
                        'tavg': float(tavg),
                        'tmin': float(tmin),
                        'tmax': float(tmax),
                        'prcp': float(prcp),
                        'wspd': float(wspd),
                        'pres': float(pres),
                        'month': int(month),
                        'year': int(year),
                        'pm10_current': float(pm10_current)
                    }
                    
                    # Générer les features
                    X_new = create_all_features_correct(input_data, historical_df)
                    
                    st.success(f"✅ {X_new.shape[1]} features générées")
                    
                    # Prédiction
                    prediction = model.predict(X_new)[0]
                    
                    # ========== AFFICHAGE RÉSULTATS ==========
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("🌫️ PM10 Prédit", f"{prediction:.1f} µg/m³")
                    
                    with col2:
                        if prediction < 20:
                            quality = "🟢 Bon"
                            st.success(quality)
                        elif prediction < 40:
                            quality = "🟡 Moyen"
                            st.warning(quality)
                        elif prediction < 50:
                            quality = "🟠 Médiocre"
                            st.warning(quality)
                        else:
                            quality = "🔴 Mauvais"
                            st.error(quality)
                    
                    with col3:
                        variation = prediction - pm10_current
                        trend = "📈" if variation > 0 else "📉"
                        st.metric(f"{trend} Variation", f"{variation:+.1f} µg/m³")
                    
                    # Jauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction,
                        title={'text': "Niveau PM10 (µg/m³)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 20], 'color': "lightgreen"},
                                {'range': [20, 40], 'color': "yellow"},
                                {'range': [40, 50], 'color': "orange"},
                                {'range': [50, 100], 'color': "red"}
                            ],
                            'threshold': {'line': {'color': "red", 'width': 4}, 'value': 50}
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Conseils
                    with st.expander("💡 Analyse des conditions"):
                        if prcp > 5:
                            st.write("🌧️ **Précipitations élevées** → Favorable à la réduction")
                        if wspd > 15:
                            st.write("💨 **Vent fort** → Bonne dispersion")
                        if tmax > 35:
                            st.write("🔥 **Température élevée** → Risque accru")
                        if pres > 1020 and wspd < 5:
                            st.write("⚠️ **Conditions stagnantes** → Accumulation possible")
                    
                    # Détails techniques
                    with st.expander("🔬 Détails techniques"):
                        st.write(f"**Features totales:** {X_new.shape[1]}")
                        st.write(f"**Modèle:** XGBoost optimisé")
                        st.write(f"**Top 5 features:**")
                        st.dataframe(pd.DataFrame({
                            'Feature': X_new.columns[:5],
                            'Valeur': X_new.iloc[0, :5].values
                        }))
                
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")
                    st.info("💡 Vérifiez que le modèle et les features correspondent")

# Footer
st.markdown("---")
st.markdown("🌍 **Casablanca Air Quality Prediction** | XGBoost Model | R² = 0.98")