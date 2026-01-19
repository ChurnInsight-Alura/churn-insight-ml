import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import lightgbm


# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---

app = FastAPI(title="ChurnInsight API", description="Microservicio de Predicción de Churn con Feature Engineering en tiempo real.")

# Cargar artefactos
print("Cargando artefactos...")

with open('artifacts/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('artifacts/champion_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('artifacts/columns.pkl', 'rb') as f:
    columns_order = pickle.load(f)
with open('artifacts/umbral_decision.pkl', 'rb') as f:
    umbral_decision = pickle.load(f)

with open('artifacts/scaler_clusters.pkl', 'rb') as f:
    scaler_clusters = pickle.load(f)
with open('artifacts/factores_clusters.pkl', 'rb') as f:
    factores_clusters = pickle.load(f)
with open('artifacts/kmeans_clusters.pkl', 'rb') as f:
    kmeans_clusters = pickle.load(f)
with open('artifacts/labels_clusters.pkl', 'rb') as f:
    etiquetas_clusters = pickle.load(f)
print("Artefactos cargados!")


COLUMNS_TO_SCALE = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
    'EstimatedSalary', 'avg_tx_amount', 'days_since_last_tx', 'tx_q1q2_rate_of_change',
    'tx_q2q3_rate_of_change', 'avg_ss_duration', 'std_ss_duration',
    'days_since_last_ss', 'ss_q1q2_rate_of_change',
    'ss_q2q3_rate_of_change', 'failed_ratio_spike_q2',
    'failed_ratio_spike_q3', 'failed_ratio_volatility'
]

# --- 2. DEFINICIÓN DE DATOS (PYDANTIC) ---

class Transaction(BaseModel):
    TransactionId: str
    TransactionDate: str # Formato esperado: "YYYY-MM-DD" o ISO
    Amount: float
    TransactionType: str

class Session(BaseModel):
    SessionId: str
    SessionDate: str     # Formato esperado: "YYYY-MM-DD"
    DurationMin: float
    FailedLogin: int     # 0 o 1

class ClientProfile(BaseModel):
    CustomerId: str
    Surname: str
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

class FullCustomerData(BaseModel):
    cliente: ClientProfile
    transacciones: List[Transaction] = []
    sesiones: List[Session] = []

# --- 3. MOTOR DE INGENIERÍA DE CARACTERÍSTICAS (La Lógica Compleja) ---

def engineer_features(data: FullCustomerData) -> pd.DataFrame:
    """
    Transforma el objeto complejo (Cliente + Transacciones + Sesiones)
    en una única fila de DataFrame lista para el modelo.
    """
    
    # A. Configurar fechas relativas (Simulando las ventanas Q1, Q2, Q3)
    # Asumimos que el corte es HOY. 
    # Q3 = Últimos 90 días
    # Q2 = Hace 90-180 días
    # Q1 = Hace 180-275 días
    cutoff_date = datetime.now()
    window_3_start = cutoff_date - timedelta(days=90)
    window_2_start = cutoff_date - timedelta(days=180)
    window_1_start = cutoff_date - timedelta(days=275)

    # ---------------- PROCESAMIENTO DE TRANSACCIONES ----------------
    if not data.transacciones:
        # Si no hay datos, creamos valores por defecto (0)
        feats_tx = {
            'avg_tx_amount': 0.0,
            'days_since_last_tx': 999, # Un número alto indicando inactividad
            'tx_q1q2_rate_of_change': 0.0,
            'tx_q2q3_rate_of_change': 0.0
        }
    else:
        # Convertir a DataFrame
        df_tx = pd.DataFrame([t.model_dump() for t in data.transacciones])
        df_tx['TransactionDate'] = pd.to_datetime(df_tx['TransactionDate'])
        
        # Filtros de Ventana
        tx_q3 = df_tx[(df_tx['TransactionDate'] >= window_3_start) & (df_tx['TransactionDate'] <= cutoff_date)]
        tx_q2 = df_tx[(df_tx['TransactionDate'] >= window_2_start) & (df_tx['TransactionDate'] < window_3_start)]
        tx_q1 = df_tx[(df_tx['TransactionDate'] >= window_1_start) & (df_tx['TransactionDate'] < window_2_start)]
        
        df_tx = pd.concat([tx_q1, tx_q2, tx_q3])  # Asegurarse de usar solo datos dentro de las ventanas

        if df_tx.empty:
            # Si después de filtrar no queda nada (cliente inactivo hace > 9 meses)
            # Asignamos valores por defecto de inactividad
            feats_tx = {
                'avg_tx_amount': 0.0,
                'days_since_last_tx': 999, # Penalización por inactividad
                'tx_q1q2_rate_of_change': 0.0,
                'tx_q2q3_rate_of_change': 0.0
            }
        else:
            # Cálculos Generales (Solo si hay datos recientes)
            last_date = df_tx['TransactionDate'].max()
            days_since = (cutoff_date - last_date).days
            
            # Totales por Q
            count_q1 = len(tx_q1)
            count_q2 = len(tx_q2)
            count_q3 = len(tx_q3)
            
            # Rates of Change
            rate_q1q2 = (count_q2 - count_q1) / (count_q1 + 1)
            rate_q2q3 = (count_q3 - count_q2) / (count_q2 + 1)
            
            feats_tx = {
                'avg_tx_amount': df_tx['Amount'].mean(),
                'days_since_last_tx': days_since,
                'tx_q1q2_rate_of_change': rate_q1q2,
                'tx_q2q3_rate_of_change': rate_q2q3
            }

    # ---------------- PROCESAMIENTO DE SESIONES ----------------
    if not data.sesiones:
        feats_ss = {
            'avg_ss_duration': 0.0,
            'std_ss_duration': 0.0,
            'days_since_last_ss': 999,
            'ss_q1q2_rate_of_change': 0.0,
            'ss_q2q3_rate_of_change': 0.0,
            'failed_ratio_spike_q2': 0.0,
            'failed_ratio_spike_q3': 0.0,
            'failed_ratio_volatility': 0.0
        }
    else:
        df_ss = pd.DataFrame([s.model_dump() for s in data.sesiones])
        df_ss['SessionDate'] = pd.to_datetime(df_ss['SessionDate'])
        
        # Filtros de Ventana
        ss_q3 = df_ss[(df_ss['SessionDate'] >= window_3_start) & (df_ss['SessionDate'] <= cutoff_date)]
        ss_q2 = df_ss[(df_ss['SessionDate'] >= window_2_start) & (df_ss['SessionDate'] < window_3_start)]
        ss_q1 = df_ss[(df_ss['SessionDate'] >= window_1_start) & (df_ss['SessionDate'] < window_2_start)]
        
        df_ss = pd.concat([ss_q1, ss_q2, ss_q3])  # Asegurarse de usar solo datos dentro de las ventanas

        if df_ss.empty:
            # Caso 2: Tiene sesiones antiguas, pero NINGUNA en los últimos 9 meses.
            # Sin este IF, el código fallaría al intentar calcular máximos o desvíos sobre vacío.
            feats_ss = {
                'avg_ss_duration': 0.0,
                'std_ss_duration': 0.0,
                'days_since_last_ss': 999, # Penalización por inactividad reciente
                'ss_q1q2_rate_of_change': 0.0,
                'ss_q2q3_rate_of_change': 0.0,
                'failed_ratio_spike_q2': 0.0,
                'failed_ratio_spike_q3': 0.0,
                'failed_ratio_volatility': 0.0
            }
        else:
            # Caso 3: Tiene actividad reciente. Procedemos con los cálculos.
            
            # Cálculos base
            last_date_ss = df_ss['SessionDate'].max()
            days_since_ss = (cutoff_date - last_date_ss).days
            
            # Desvío estándar (Controlando si hay 1 solo dato)
            std_duration = df_ss['DurationMin'].std()
            if pd.isna(std_duration): std_duration = 0.0
            
            # Lógica Rate of Change
            c_q1 = len(ss_q1)
            c_q2 = len(ss_q2)
            c_q3 = len(ss_q3)
            
            def calc_rate(past, current):
                if past == 0: return 1.0 if current > 0 else 0.0
                return (current - past) / past

            rate_ss_q1q2 = calc_rate(c_q1, c_q2)
            rate_ss_q2q3 = calc_rate(c_q2, c_q3)
            
            # Lógica Failed Ratios (Intentos fallidos)
            def get_ratio(df_subset):
                total = len(df_subset)
                if total == 0: return 0.0
                failed = df_subset['FailedLogin'].sum()
                return failed / total
                
            ratio_q1 = get_ratio(ss_q1)
            ratio_q2 = get_ratio(ss_q2)
            ratio_q3 = get_ratio(ss_q3)
            
            spike_q2 = ratio_q2 - ratio_q1
            spike_q3 = ratio_q3 - ratio_q2
            volatility = np.std([ratio_q1, ratio_q2, ratio_q3])
            
            feats_ss = {
                'avg_ss_duration': df_ss['DurationMin'].mean(),
                'std_ss_duration': std_duration,
                'days_since_last_ss': days_since_ss,
                'ss_q1q2_rate_of_change': rate_ss_q1q2,
                'ss_q2q3_rate_of_change': rate_ss_q2q3,
                'failed_ratio_spike_q2': spike_q2,
                'failed_ratio_spike_q3': spike_q3,
                'failed_ratio_volatility': volatility
            }

    # ---------------- ENSAMBLAJE FINAL ----------------
    # Datos base del cliente
    base_data = data.cliente.model_dump()
    
    # Combinar diccionarios
    full_row = {**base_data, **feats_tx, **feats_ss}
    
    # Crear DataFrame
    df_final = pd.DataFrame([full_row])
    
    # Importante: Llenar Nulos si quedaron (ej: avg duration si no hubo sesiones)
    df_final = df_final.fillna(0)
    
    return df_final


def get_customer_segment(row_data: dict) -> tuple:
    """
    Calcula el segmento del cliente replicando la lógica manual + KMeans.
    Versión 'Purista' usando DataFrames para mantener nombres de features.
    """
    # 1. Extraer variables necesarias
    tenure = row_data.get('Tenure', 0)
    age = row_data.get('Age', 0)
    balance = row_data.get('Balance', 0.0)
    est_salary = row_data.get('EstimatedSalary', 0.0)
    credit_score = row_data.get('CreditScore', 0)
    num_products = row_data.get('NumOfProducts', 0)
    has_crcard = row_data.get('HasCrCard', 0)
    is_active = row_data.get('IsActiveMember', 0)

    # 2. Calcular Scores "Crudos"
    wealth_score = (balance * factores_clusters['Balance']) + \
                   (est_salary * factores_clusters['EstimatedSalary']) + \
                   (credit_score * factores_clusters['CreditScore'])
    
    engagement_score = (num_products * factores_clusters['NumOfProducts']) + \
                       (has_crcard * factores_clusters['HasCrCard']) + \
                       (is_active * factores_clusters['IsActiveMember'])

    raw_time_score = (tenure * factores_clusters['Tenure']) + \
                     (age * factores_clusters['Age'])

    # --- CORRECCIÓN PURISTA PARTE 1: SCALER ---
    # Creamos un DataFrame temporal solo para el Scaler.
    # IMPORTANTE: Los nombres 'WealthScore' y 'EngagementScore' deben coincidir
    # con los que usaste en el entrenamiento. Si allá se llamaban "w_score", cámbialo aquí.
    df_to_scale = pd.DataFrame(
        [[wealth_score, engagement_score]], 
        columns=['wealth_score', 'engagement_score'] 
    )
    
    # Ahora el scaler recibe un DataFrame con nombres, no se quejará.
    scores_norm = scaler_clusters.transform(df_to_scale)
    
    wealth_norm = scores_norm[0][0]
    engagement_norm = scores_norm[0][1]

    # 4. Lógica Cluster Time
    cluster_time_num = 0.5 
    if raw_time_score < 15: 
        cluster_time_num = 0.0
    elif raw_time_score > 20:
        cluster_time_num = 1.0

    # 5. Calcular Score Final
    final_score = (factores_clusters['ClusterWealth'] * wealth_norm) + \
                  (factores_clusters['ClusterEngagement'] * engagement_norm) + \
                  (factores_clusters['ClusterTime'] * cluster_time_num)

    # 6. Predecir Cluster con KMeans
    # Creamos un DataFrame para la predicción final.
    # El nombre de columna debe coincidir con el del entrenamiento.
    df_for_kmeans = pd.DataFrame(
        [[final_score]], 
        columns=['final_score'] 
    )

    # Pasamos el DataFrame entero, no el valor suelto
    cluster_id = kmeans_clusters.predict(df_for_kmeans)[0]

    # 7. Mapear a etiqueta
    segment_name = etiquetas_clusters.get(cluster_id, "Desconocido")
    
    return segment_name

def calculate_priority(churn_prob: float, segment: str) -> str:
    """
    Define la prioridad de acción cruzando Probabilidad de Churn vs Segmento
    """
    # Si la probabilidad de irse es muy baja, no molestamos al cliente
    if churn_prob < umbral_decision:
        return "Baja - Mantener Contento"
    
    # Si la probabilidad es ALTA, vemos quién es el cliente
    if churn_prob > 0.7:
        if segment in ['VIP', 'Valioso - Bajo compromiso']:
            return "CRÍTICO - Llamar Inmediatamente" # Es valioso y se va a ir
        elif segment in ['Cliente potencial', 'Standard']:
            return "Alta - Ofrecer Incentivo"
        else:
            return "Media - Correo Electrónico Automático" # Es 'Poco Valor', no gastamos recursos caros
            
    # Zona gris (probabilidad media [umbral de decision - 0.7])
    if segment == 'VIP':
        return "Alta - Chequeo Personalizado"
        
    return "Media - Monitorear"

@app.post("/predict/customer")
def predict_single_customer(data: FullCustomerData):
    """Predicción para un solo cliente con estructura compleja."""
    
    # 1. FEATURE ENGINEERING
    df = engineer_features(data)
    
    cols_to_drop = ['Surname', 'CustomerId']
    # Usamos errors='ignore' por si alguna no existe, para que no falle.
    df_model_input = df.drop(columns=cols_to_drop, errors='ignore')

    # 2. ENCODING
    # El encoder necesita recibir TODAS las columnas originales
    # Nota: ColumnTransformer espera nombres específicos, asegurse que entren
    try:
        # make_column_transformer devuelve un array numpy, no un DF
        X_encoded_array = encoder.transform(df)
        
        # Recuperar los nombres de las columnas para poder filtrar
        # las que se quiere escalar vs las que no.
        feature_names_out = encoder.get_feature_names_out()
        
        # Limpiar los nombres de las columnas (quitar prefijos)
        clean_names = [col.split('__')[-1] for col in feature_names_out]
        
        # Reconstruir DataFrame encoded
        df_encoded = pd.DataFrame(X_encoded_array, columns=clean_names)
        
    except Exception as e:
        return {"error": f"Error en Encoding: {str(e)}"}

    # 3. SCALING
    try:
        # A. Separar las numéricas que requieren scaling
        # Asegurarse que solo se buscan las columnas que existan en el df_encoded
        cols_to_scale_present = [c for c in COLUMNS_TO_SCALE if c in df_encoded.columns]
        
        df_to_scale = df_encoded[cols_to_scale_present]
        df_binary = df_encoded.drop(columns=cols_to_scale_present)
        
        # B. Escalamos solo las numéricas
        X_scaled_array = scaler.transform(df_to_scale)
        df_scaled_part = pd.DataFrame(X_scaled_array, columns=cols_to_scale_present, index=df_encoded.index)
        
        # C. Concatenamos (Unimos Scaled + Binary)
        df_final_processed = pd.concat([df_scaled_part, df_binary], axis=1)
        
        # D. REORDENAMIENTO FINAL (CRUCIAL)
        # Usamos columns_order para dejarlo idéntico al entrenamiento
        df_ready = df_final_processed[columns_order]
        
    except Exception as e:
        return {"error": f"Error en Scaling/Reordering: {str(e)}"}
    
    # 4. ChurnProbability y ChurnPredicción
    prob = model.predict_proba(df_ready)[0][1]
    prediction = int(prob >= umbral_decision)
    
    cliente_dict = data.cliente.model_dump()
    
    # CustomerSegment
    customer_segment = get_customer_segment(cliente_dict)
    
    # InterventionPriority
    priority = calculate_priority(prob, customer_segment)

    
    return {
        "CustomerId": data.cliente.CustomerId,
        "PredictedProba": (round(prob, 4)) * 100,
        "PredictedLabel": prediction,
        "CustomerSegment": customer_segment,
        "InterventionPriority": priority
    }

@app.post("/predict/batch")
def predict_batch(batch_data: List[FullCustomerData]):
    """Predicción masiva para una lista de clientes."""
    results = []
    for customer in batch_data:
        # Reutiliza la lógica individual
        res = predict_single_customer(customer)
        results.append(res)

    
    return results



@app.post("/predict/batch_stats")
def predict_batch_stats(batch_data: List[FullCustomerData]):
    """
    Predicción masiva que devuelve estadísticas de negocio, financieras y de tendencia.
    """
    processed_rows = []

    # 1. Iteramos sobre el batch para construir un DataFrame maestro
    for data in batch_data:
        # A. Predicción
        pred_res = predict_single_customer(data)
        
        # B. Feature Engineering
        feats_df = engineer_features(data)
        
        # C. Armamos la fila combinada
        row = {
            'PredictedLabel': pred_res['PredictedLabel'],
            'InterventionPriority': pred_res['InterventionPriority'],
            'Geography': data.cliente.Geography,
            'IsActiveMember': data.cliente.IsActiveMember,
            'Balance': data.cliente.Balance,
            # Variables de Tiempo
            'days_since_last_tx': feats_df['days_since_last_tx'].values[0],
            'days_since_last_ss': feats_df['days_since_last_ss'].values[0],
            # Variables de Tendencia
            'tx_q2q3_rate_of_change': feats_df['tx_q2q3_rate_of_change'].values[0],
            'ss_q2q3_rate_of_change': feats_df['ss_q2q3_rate_of_change'].values[0]
        }
        processed_rows.append(row)

    # 2. Convertimos a DataFrame
    df = pd.DataFrame(processed_rows)

    if df.empty:
        return {"error": "El batch está vacío"}

    # 3. Cálculos Estadísticos
    
    total_clients = len(df)
    priority_counts = df['InterventionPriority'].value_counts().to_dict()
    
    # Filtro de Churners (La base de todos los riesgos)
    churners = df[df['PredictedLabel'] == 1]

    # -- Cálculos de Seguridad (Std y Nans) --
    def safe_std(series):
        val = series.std()
        return 0.0 if pd.isna(val) else val

    # -- Lógica Donut Chart --
    qty_low = int(priority_counts.get("Baja - Mantener Contento", 0))
    qty_red_donut = total_clients - qty_low
    
    # -- Lógica Churn Rate --
    churn_count = len(churners)
    churn_prop = (churn_count / total_clients) * 100 if total_clients > 0 else 0.0

    # --- NUEVA LÓGICA: CRUCE ACTIVIDAD vs PREDICCIÓN ---
    # Active (1) & No se va (0)
    act_exit_0 = len(df[(df['IsActiveMember'] == 1) & (df['PredictedLabel'] == 0)])
    # Active (1) & Se va (1) -> ¡PELIGROSO!
    act_exit_1 = len(df[(df['IsActiveMember'] == 1) & (df['PredictedLabel'] == 1)])
    
    # Inactive (0) & No se va (0)
    inact_exit_0 = len(df[(df['IsActiveMember'] == 0) & (df['PredictedLabel'] == 0)])
    # Inactive (0) & Se va (1) -> ¡LO ESPERADO!
    inact_exit_1 = len(df[(df['IsActiveMember'] == 0) & (df['PredictedLabel'] == 1)])


    # 4. Construcción del JSON
    stats = {
        # --- KPIs FINANCIEROS Y DE NEGOCIO ---
        'ChurnersTotalBalance': float(round(churners['Balance'].sum(), 2)),
        'DonutGreen': qty_low,
        'DonutRed': qty_red_donut,
        'CustomerChurnProp': float(round(churn_prop, 2)),

        # --- DETALLE DE PRIORIDADES (Embudo) ---
        'QtyCritical': int(priority_counts.get("CRÍTICO - Llamar Inmediatamente", 0)),
        'QtyHigh-Incentive': int(priority_counts.get("Alta - Ofrecer Incentivo", 0)),
        'QtyHigh-Personalized': int(priority_counts.get("Alta - Chequeo Personalizado", 0)),
        'QtyMedium-Monitor': int(priority_counts.get("Media - Monitorear", 0)),
        'QtyMedium-Mail': int(priority_counts.get("Media - Correo Electrónico Automático", 0)),
        'QtyLow': qty_low,
        
        # --- MATRIZ DE ACTIVIDAD ---
        'QtyIsActiveMemberExit_0': int(act_exit_0),
        'QtyIsActiveMemberExit_1': int(act_exit_1),
        'QtyIsNOTActiveMemberExit_0': int(inact_exit_0),
        'QtyIsNOTActiveMemberExit_1': int(inact_exit_1),

        # --- ESTADÍSTICAS GEO ---
        'ChurnersGermany': int(len(churners[churners['Geography'] == 'Germany'])),
        'ChurnersSpain': int(len(churners[churners['Geography'] == 'Spain'])),
        'ChurnersFrance': int(len(churners[churners['Geography'] == 'France'])),

        # --- SECCIÓN TRANSACCIONES (KPIs + Velocímetro) ---
        'AvgTxDaysSinceLast': float(round(df['days_since_last_tx'].mean(), 2)),
        'StdTxDaysSinceLast': float(round(safe_std(df['days_since_last_tx']), 2)),
        'AvgTxRateChange': float(round(df['tx_q2q3_rate_of_change'].mean(), 4)),

        # --- SECCIÓN SESIONES (KPIs + Velocímetro) ---
        'AvgSsDaysSinceLast': float(round(df['days_since_last_ss'].mean(), 2)),
        'StdSsDaysSinceLast': float(round(safe_std(df['days_since_last_ss']), 2)),
        'AvgSsRateChange': float(round(df['ss_q2q3_rate_of_change'].mean(), 4))
    }

    return stats

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
