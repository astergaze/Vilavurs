import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#model random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
# linear model
# from sklearn.linear_model import LogisticRegression
df = None 
try:
    df = pd.read_csv(
        "./data/cumulative_2025.10.04_10.14.29.csv", 
        sep=',',
        comment='#' 
    )
    # print(df.head())
    
except Exception as e:
    print(f"There was an error when loading the file: {e}")
if df is not None: 
    #'koi_disposition'
    features = ['koi_period', 'koi_duration', 'koi_prad', 'koi_teq', 'koi_steff', 'koi_srad', 'koi_smass']
    # print(df['koi_disposition'].value_counts()) # Confirmed, candidate, false positive
    # print(df['koi_period'].value_counts()) # Orbital period (days)
    # print(df['koi_duration'].value_counts()) # Transit duration (hours)
    # print(df['koi_prad'].value_counts()) # Planet radio
    # print(df['koi_teq'].value_counts()) # Temperature of equilibrium
    # print(df['koi_steff'].value_counts()) # K temperature
    # print(df['koi_srad'].value_counts()) # Stellar Radio
    # print(df['koi_smass'].value_counts()) # Stellar Mass
    # print(df[features].isnull().sum())
    # print("###############################################################################")
    for col in features:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
    # print(df[features].isnull().sum())
    # print("###############################################################################")
    df['target'] = df['koi_disposition'].map({
        'CONFIRMED': 2,
        'CANDIDATE': 1,
        'FALSE POSITIVE': 0
    })
    df.dropna(subset=['target'], inplace=True)
else: 
    print("There was an error, check the comments or the separation of the file")
X = df[features]
y = df['target']
# Usamos stratify=y para asegurar una distribución similar de clases en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y 
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#random forest test
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) 
y_pred_rf = rf_model.predict(X_test)
# print("--- Random Forest Report ---") #accuracy 73.6%
# print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}") 
# print(classification_report(y_test, y_pred_rf))
# #linear test, didn't win
# lr_model = LogisticRegression(max_iter=1000, random_state=42)
# lr_model.fit(X_train_scaled, y_train)
# y_pred_lr = lr_model.predict(X_test_scaled)
# print("--- Logistic Regression Report ---")
# print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
# print(classification_report(y_test, y_pred_lr))
importance = rf_model.feature_importances_
for i, score in enumerate(importance):
    print(f'{features[i]}: {score:.4f}')
param_grid = {
    'n_estimators': [100, 200],  # Número de árboles
    'max_depth': [5, 10, None]   # Profundidad máxima
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# El mejor modelo es:
best_model = grid_search.best_estimator_
print(f"Mejores parámetros: {grid_search.best_params_}")
joblib.dump(best_model, 'best_exoplanet_model.pkl')
joblib.dump(scaler, 'scaler.pkl') 