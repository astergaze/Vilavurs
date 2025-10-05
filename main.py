import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer 
# NUEVO MODELO: Random Forest (Mejor para desbalance sin SMOTE y datos ruidosos)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- RUTAS DE ARCHIVOS DE SALIDA E ENTRADA ---
OUTPUT_MODEL_PATH = './best_exoplanet_model.pkl'
OUTPUT_SCALER_PATH = './scaler.pkl'
OUTPUT_IMPUTER_PATH = './imputer.pkl' 
# Rutas de los tres datasets individuales (requeridos)
KEPLER_PATH = "./data/cumulative_2025.10.04_10.14.29.csv" 
TESS_PATH = "./data/TOI_2025.10.04_14.28.52.csv"
K2_PATH = "./data/k2pandc_2025.10.04_14.29.09.csv"

# --- LISTA DE FEATURES (15 COLUMNAS) ---
KEPLER_FEATURES = [
    'koi_period', 'koi_duration', 'koi_prad', 'koi_teq', 'koi_steff', 'koi_srad', 'koi_smass',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 
    'koi_depth', 'koi_slogg',
    'koi_model_snr', 'koi_ror'
]

# --- MAPEO DE COLUMNAS (Para armonizar datos K2/TESS a Kepler) ---
COLUMN_MAP = {
    'pl_orbper': 'koi_period',       
    'pl_trandurh': 'koi_duration',   
    'pl_rade': 'koi_prad',           
    'pl_eqt': 'koi_teq',             
    'st_teff': 'koi_steff',          
    'st_rad': 'koi_srad',            
    'st_mass': 'koi_smass',          
    'pl_trandep': 'koi_depth',       
    'st_logg': 'koi_slogg',          
    'pl_ratror': 'koi_ror',          
    'pl_trandur': 'koi_duration', 
}

# --- MAPEO DE CLASES (Target Y) ---
DISPOSITIONS_TO_MAP = {
    'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0, 'FALSE_POSITIVE': 0,
    # TESS
    'CP': 2, 'PC': 1, 'KP': 1, 'FP': 0,
}

# --- FUNCIÓN CENTRAL DE ARMONIZACIÓN Y PRE-PROCESAMIENTO ---
def preprocess_data(df):
    """Armoniza, renombra, calcula ROR, convierte unidades K2, e imputa NaN."""
    
    df_processed = df.copy()
    
    # 1. Renombrar y Estandarizar Nombres
    df_processed.rename(columns=COLUMN_MAP, inplace=True)
    
    # 2. Identificación y Conversión de K2 (koi_depth de % a ppm)
    is_k2_source = 'disposition' in df_processed.columns and 'koi_disposition' not in df_processed.columns
    if is_k2_source and 'koi_depth' in df_processed.columns:
        df_processed['koi_depth'] = df_processed['koi_depth'] * 10000
    
    # 3. Cálculo de koi_ror (si es necesario)
    if 'koi_ror' not in df_processed.columns and 'koi_prad' in df_processed.columns and 'koi_srad' in df_processed.columns:
        R_EARTH_TO_R_SUN = 0.009158
        df_processed['koi_ror'] = np.where(
            (df_processed['koi_srad'].isnull()) | (df_processed['koi_srad'] == 0),
            np.nan,
            df_processed['koi_prad'] * R_EARTH_TO_R_SUN / df_processed['koi_srad']
        )
        
    # 4. Imputación de Features Faltantes Específicas
    for col in KEPLER_FEATURES:
        if col not in df_processed.columns:
            if col.startswith('koi_fpflag_'):
                df_processed[col] = 0
            else:
                df_processed[col] = np.nan
            
    # 5. Mapeo de la Variable Target (Y) y limpieza
    target_col = None
    for col in ['koi_disposition', 'disposition', 'tfopwg_disp']:
        if col in df_processed.columns:
            target_col = col
            break
            
    if target_col is None:
        return None, None
        
    df_processed['y_class'] = df_processed[target_col].astype(str).str.upper().str.strip().replace(' ', '_', regex=True).map(DISPOSITIONS_TO_MAP)
    
    # Seleccionar features finales
    df_final = df_processed[KEPLER_FEATURES + ['y_class']].copy()
    
    # Eliminamos filas donde falta la etiqueta (y_class) para el entrenamiento
    df_final.dropna(subset=['y_class'], inplace=True)
    df_final['y_class'] = df_final['y_class'].astype(int)
    
    return df_final, target_col

def load_and_preprocess_single_set(path, name):
    """Carga un dataset individual y aplica el preprocesamiento."""
    try:
        logging.info(f"Cargando y preprocesando datos de {name} desde: {path}")
        df_raw = pd.read_csv(path, comment='#', low_memory=False)
        df_processed, _ = preprocess_data(df_raw)
        
        if df_processed is None:
            logging.error(f"Fallo en el preprocesamiento de {name}. El archivo podría estar corrupto o faltar la columna de disposición.")
            return None
        
        logging.info(f"{name} procesado: {len(df_processed)} filas.")
        return df_processed
        
    except FileNotFoundError:
        logging.warning(f"ADVERTENCIA: Archivo de {name} no encontrado en: {path}. Este set será omitido.")
        return None
    except Exception as e:
        logging.error(f"Error al cargar o preprocesar el CSV de {name}: {e}")
        return None

# --- FUNCIÓN PRINCIPAL DE ENTRENAMIENTO CON RANDOM FOREST ---
def train_random_forest_with_mice():
    """
    Ejecuta el flujo completo de carga, imputación MICE, escalado y 
    entrenamiento de Random Forest con class_weight='balanced'.
    """
    
    # 1. Cargar y preprocesar los tres datasets individualmente
    df_kepler = load_and_preprocess_single_set(KEPLER_PATH, "Kepler")
    df_tess = load_and_preprocess_single_set(TESS_PATH, "TESS")
    df_k2 = load_and_preprocess_single_set(K2_PATH, "K2")
    
    # 2. Concatenar todos los DataFrames
    all_data = [df for df in [df_kepler, df_tess, df_k2] if df is not None]
    
    if not all_data:
        logging.error("No se pudieron cargar ni procesar datos válidos. Abortando el entrenamiento.")
        return

    df_processed = pd.concat(all_data, ignore_index=True)
    logging.info(f"Datasets combinados. Total de filas para entrenamiento: {len(df_processed)}")
    
    X_full = df_processed[KEPLER_FEATURES]
    y_full = df_processed['y_class']
    
    # 3. División de Datos (Entrenamiento y Prueba)
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    
    # 4. Imputación de Faltantes (USANDO IterativeImputer/MICE)
    logging.info("Aplicando Imputador Iterativo (MICE) para estimar valores faltantes...")
    imputer = IterativeImputer(random_state=42, max_iter=20) 
    
    X_train_imputed_array = imputer.fit_transform(X_train_orig)
    X_train_imputed = pd.DataFrame(X_train_imputed_array, columns=X_train_orig.columns)
    
    X_test_imputed_array = imputer.transform(X_test_orig)
    X_test_imputed = pd.DataFrame(X_test_imputed_array, columns=X_test_orig.columns)
    
    logging.info(f"Imputación iterativa completada. Valores NaN eliminados.")

    # 5. Escalado de Datos (Estándar)
    # El escalado no es estrictamente necesario para Random Forest, pero se mantiene 
    # para estandarizar el pipeline y si se quisiera probar otro modelo después.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    X_train_final = pd.DataFrame(X_train_scaled, columns=KEPLER_FEATURES)
    X_test_final = pd.DataFrame(X_test_scaled, columns=KEPLER_FEATURES)
    y_train_final = y_train
    
    # 6. Entrenamiento del Modelo (Random Forest con class_weight='balanced')
    logging.info("\n--- Entrenando Random Forest con MICE y class_weight='balanced' ---")
    
    # Ajuste de hiperparámetros para mejorar el rendimiento
    best_model = RandomForestClassifier(
        n_estimators=200,             # Más árboles
        max_depth=30,                 # Profundidad limitada para evitar sobreajuste extremo
        min_samples_split=2,
        class_weight='balanced',      # CRÍTICO: Pondera las clases minoritarias
        random_state=42,
        n_jobs=-1,                    # Usar todos los núcleos disponibles
        verbose=1
    )
    
    best_model.fit(X_train_final, y_train_final)
    
    logging.info("Modelo Random Forest entrenado.")

    # 7. Evaluación del Modelo Final en el Conjunto de PRUEBA
    y_pred_test = best_model.predict(X_test_final)
    accuracy = accuracy_score(y_test, y_pred_test)
    
    logging.info("\n--- Reporte del Modelo FINAL OPTIMIZADO (Random Forest + MICE) ---")
    logging.info(f"Precisión (Accuracy) en el set de prueba (General): {accuracy*100:.2f}%")
    
    # Imprimir el reporte de clasificación completo
    report_lines = classification_report(y_test, y_pred_test, output_dict=False)
    for line in report_lines.split('\n'):
        logging.info(line)
    
    # 8. Guardar Modelo, Scaler e Imputer
    joblib.dump(best_model, OUTPUT_MODEL_PATH)
    joblib.dump(scaler, OUTPUT_SCALER_PATH)
    joblib.dump(imputer, OUTPUT_IMPUTER_PATH)
    logging.info(f"\nModelo FINAL optimizado guardado en: {OUTPUT_MODEL_PATH}")
    logging.info(f"Escalador guardado en: {OUTPUT_SCALER_PATH}")
    logging.info(f"Imputador MICE guardado en: {OUTPUT_IMPUTER_PATH}")
    
    logging.info("¡El proceso de reentrenamiento ha finalizado! Prueba este nuevo modelo, es el que mejor compensará el desbalance sin usar SMOTE.")

if __name__ == "__main__":
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        logging.error("Faltan librerías. Ejecuta: pip install scikit-learn")
    else:
        train_random_forest_with_mice()