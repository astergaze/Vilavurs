import streamlit as st
import joblib
import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import confusion_matrix, accuracy_score

MODEL_PATH = './best_exoplanet_model.pkl'
SCALER_PATH = './scaler.pkl'
IMPUTER_PATH = './imputer.pkl'

model = None
scaler = None
imputer = None

st.set_page_config(
    page_title="Vilavurs: Exoplanet Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

KEPLER_FEATURES = [
    'koi_period', 'koi_duration', 'koi_prad', 'koi_teq', 'koi_steff', 'koi_srad', 'koi_smass',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 
    'koi_depth', 'koi_slogg',
    'koi_model_snr', 'koi_ror'
]

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
}

@st.cache_data
def harmonize_batch_data(df):
    df_harmonized = df.copy()
    
    is_k2_source = 'disposition' in df_harmonized.columns and 'koi_disposition' not in df_harmonized.columns

    df_harmonized.rename(columns=COLUMN_MAP, inplace=True)

    if is_k2_source and 'koi_depth' in df_harmonized.columns:
        df_harmonized['koi_depth'] = df_harmonized['koi_depth'] * 10000 
        st.info("K2 file detected: 'koi_depth' column converted from % to ppm (x10000).")
    
    if 'koi_ror' not in df_harmonized.columns and 'koi_prad' in df_harmonized.columns and 'koi_srad' in df_harmonized.columns:
        R_EARTH_TO_R_SUN = 0.009158
        df_harmonized['koi_ror'] = np.where(
            (df_harmonized['koi_srad'].isnull()) | (df_harmonized['koi_srad'] == 0),
            np.nan, 
            df_harmonized['koi_prad'] * R_EARTH_TO_R_SUN / df_harmonized['koi_srad']
        )
        st.info("'koi_ror' column calculated from planetary and stellar radii.")

    for col in KEPLER_FEATURES:
        if col not in df_harmonized.columns:
            if col.startswith('koi_fpflag_'):
                df_harmonized[col] = 0
            else:
                df_harmonized[col] = np.nan
            
    df_harmonized = df_harmonized[KEPLER_FEATURES].copy()
    df_harmonized = df_harmonized.astype(float) 
    
    return df_harmonized


try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH) 
    # Línea st.sidebar.success("Model, scaler, and MICE imputer loaded.") eliminada
except FileNotFoundError:
    st.sidebar.error("Error: Model, scaler, or imputer file not found.")
    st.warning("The application cannot perform predictions. Please verify the existence of the .pkl files.")
except Exception as e:
    st.sidebar.error(f"Critical error: {e}")


def map_prediction(prediction):
    if prediction == 2:
        return "Confirmed Exoplanet", "Confirmed"
    elif prediction == 1:
        return "Planetary Candidate", "Candidate"
    else:
        return "False Positive", "False Positive"

def map_prediction_raw(prediction_raw):
    mapping = {2: 'Confirmed', 1: 'Candidate', 0: 'False Positive'}
    return pd.Series(prediction_raw).map(mapping)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')
    
st.title("Vilavurs: AI Exoplanet Classifier")
st.markdown("Trained with harmonized data from **Kepler, TESS, and K2**.")

# --- MODE SELECTION IN SIDEBAR ---
st.sidebar.write("---")
st.sidebar.title("Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Choose your input method:",
    ["Batch Analysis (Upload File)", "Single Case Analysis (Manual Input)"],
    index=0,
)
# --- END SIDEBAR MODE SELECTION ---

# Initialize uploaded_file outside the main if/elif block
uploaded_file = None

if analysis_mode == "Batch Analysis (Upload File)":
    # Moviendo el Input al Sidebar
    st.sidebar.write("---")
    st.sidebar.subheader("File Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a Kepler, TESS, or K2 CSV file to classify multiple objects.", 
        type=["csv"],
        help="The application will automatically harmonize TESS/K2 column names and units to the Kepler format for prediction."
    )
    
    
    st.header("Batch Analysis Results")
    
    if uploaded_file is not None:
        if model is None or scaler is None or imputer is None:
            st.warning("The model, scaler, or imputer is unavailable. Batch analysis cannot be performed.")
        else:
            try:
                batch_df = pd.read_csv(uploaded_file, comment='#', low_memory=False)
                
                st.info(f"File uploaded with **{len(batch_df)}** rows. Starting data harmonization process...")

                original_df_for_context = batch_df.copy()
                X_batch_harmonized = harmonize_batch_data(batch_df)
                
                with st.spinner("Applying MICE Imputer and scaling..."):
                    X_batch_imputed_array = imputer.transform(X_batch_harmonized)
                    X_batch = pd.DataFrame(X_batch_imputed_array, columns=KEPLER_FEATURES)
                    
                    X_batch_scaled = scaler.transform(X_batch.values) 
                    X_batch_scaled_df = pd.DataFrame(X_batch_scaled, columns=KEPLER_FEATURES)
                    
                    batch_predictions_raw = model.predict(X_batch_scaled_df)
                    batch_probabilities = model.predict_proba(X_batch_scaled_df)
                
                results_df = X_batch.copy()
                results_df['Classification_Num'] = batch_predictions_raw 
                results_df['Classification'] = map_prediction_raw(batch_predictions_raw)
                results_df['Prob_FP'] = [p[0] * 100 for p in batch_probabilities]
                results_df['Prob_Cand'] = [p[1] * 100 for p in batch_probabilities]
                results_df['Prob_Conf'] = [p[2] * 100 for p in batch_probabilities]
                
                results_to_display = results_df[['Classification', 'Prob_Conf', 'Prob_Cand', 'Prob_FP', 
                                                 'koi_period', 'koi_duration', 'koi_prad', 'koi_teq', 
                                                 'koi_model_snr']]
                
                st.subheader("Results Table")
                
                st.dataframe(results_to_display.style.format({
                    'Prob_FP': "{:.2f}%", 
                    'Prob_Cand': "{:.2f}%", 
                    'Prob_Conf': "{:.2f}%",
                    'koi_period': "{:.4f}", 'koi_duration': "{:.4f}", 'koi_prad': "{:.4f}", 
                    'koi_teq': "{:.2f}", 'koi_model_snr': "{:.2f}"
                }), use_container_width=True)
                
                csv_data = convert_df_to_csv(results_to_display)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_data,
                    file_name='exoplanet_batch_predictions.csv',
                    mime='text/csv',
                    help="Download the model's classifications and probabilities."
                )

                st.write("---")
                
                st.subheader("Distribution of Predicted Classifications")
                
                counts_df = results_df['Classification'].value_counts().reset_index()
                counts_df.columns = ['Classification', 'Total']

                color_map = {
                    'False Positive': '#F44336', 
                    'Candidate': '#FFC107', 
                    'Confirmed': '#4CAF50'
                }
                
                chart = alt.Chart(counts_df).mark_bar().encode(
                    x=alt.X('Total:Q', title='Object Count'),
                    y=alt.Y('Classification:N', sort='-x', title='Predicted Class'),
                    color=alt.Color(
                        'Classification:N', 
                        scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())), 
                        legend=None
                    ),
                    tooltip=['Classification', 'Total']
                ).properties(
                    height=250,
                    title='Count of Predictions by Class in Batch'
                )
                st.altair_chart(chart, use_container_width=True)
                
                st.write("---")
                
                ground_truth_col = None
                if 'koi_disposition' in original_df_for_context.columns:
                    ground_truth_col = 'koi_disposition'
                elif 'disposition' in original_df_for_context.columns:
                    ground_truth_col = 'disposition' 
                elif 'tfopwg_disp' in original_df_for_context.columns:
                    ground_truth_col = 'tfopwg_disp'

                if ground_truth_col is not None:
                    st.subheader("Model Accuracy Verification (If file contains 'Ground Truth')")

                    dispositions_to_map = {
                        'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0, 'FALSE_POSITIVE': 0,
                        'CP': 2, 'PC': 1, 'KP': 1, 'FP': 0,
                    }
                    
                    y_true_raw = original_df_for_context[ground_truth_col].astype(str).str.upper().replace(' ', '_', regex=True).map(dispositions_to_map)
                    
                    comparison_df = pd.DataFrame({
                        'y_true': y_true_raw.reset_index(drop=True), 
                        'y_pred': results_df['Classification_Num'].reset_index(drop=True)
                    }).dropna()
                    
                    y_true_final = comparison_df['y_true'].astype(int).values
                    y_pred_final = comparison_df['y_pred'].astype(int).values

                    if len(y_true_final) > 0:
                        
                        accuracy = accuracy_score(y_true_final, y_pred_final)
                        
                        col_acc, col_total = st.columns([1, 2])
                        with col_acc:
                            st.metric(
                                label="Overall Accuracy", 
                                value=f"{accuracy*100:.2f}%", 
                                help="Percentage of correct predictions compared to the Kepler/K2/TESS disposition."
                            )
                        with col_total:
                            st.markdown(f"**Evaluated Cases:** {len(y_true_final)} (Rows with valid disposition label)")

                        cm = confusion_matrix(y_true_final, y_pred_final, labels=[0, 1, 2])
                        
                        cm_df = pd.DataFrame(cm, 
                                             index=['True FP (0)', 'True Cand (1)', 'True Conf (2)'], 
                                             columns=['Prediction FP (0)', 'Prediction Cand (1)', 'Prediction Conf (2)'])
                                                 
                        st.markdown("##### Confusion Matrix")
                        st.dataframe(cm_df, use_container_width=True)
                        st.markdown("_Row: True Value (Ground Truth) | Column: Predicted Value by Model_")
                        
                    else:
                        st.warning("Could not find rows with valid labels to verify accuracy (Ground Truth).")

                else:
                    st.warning("No disposition columns ('koi_disposition', 'disposition', 'tfopwg_disp') found in the uploaded file to verify model accuracy.")
                
                st.success("Batch analysis completed successfully.")
                
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}. Try checking the separator (is it ',' or ';') or the file encoding.")

    else:
        # Show a message when in Batch mode but no file is uploaded
        st.info("Upload a CSV file using the control in the **left sidebar** to view batch analysis results here.")


elif analysis_mode == "Single Case Analysis (Manual Input)":
    st.header("Single Case Analysis: Manual Parameter Input")
    
    with st.form("exoplanet_input_form"):
        
        st.subheader("Physical and Transit Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            period = st.number_input(r"1. Orbital Period ($koi\_period$)", min_value=0.0, value=15.0, format="%.4f", help="Orbital period in days.")
            duration = st.number_input(r"2. Transit Duration ($koi\_duration$)", min_value=0.0, value=5.0, format="%.4f", help="Transit duration in hours.")
            prad = st.number_input(r"3. Planetary Radius ($koi\_prad$)", min_value=0.0, value=2.5, format="%.4f", help="Planetary radius in Earth radii.")
            teq = st.number_input(r"4. Equilibrium Temperature ($koi\_teq$)", min_value=0.0, value=750.0, format="%.2f", help="Planetary equilibrium temperature in Kelvin (K).")
            steff = st.number_input(r"5. Stellar Temperature ($koi\_steff$)", min_value=0.0, value=5700.0, format="%.2f", help="Effective stellar temperature in Kelvin (K).")
            depth = st.number_input(r"8. Transit Depth ($koi\_depth$)", min_value=0.0, value=150.0, format="%.2f", help="Drop in stellar light flux during transit (in parts per million - ppm).")
            
        with col2:
            srad = st.number_input(r"6. Stellar Radius ($koi\_srad$)", min_value=0.0, value=1.0, format="%.4f", help="Stellar radius in Solar radii.")
            smass = st.number_input(r"7. Stellar Mass ($koi\_smass$)", min_value=0.0, value=1.0, format="%.4f", help="Stellar mass in Solar masses.")
            slogg = st.number_input(r"9. Stellar Gravity ($koi\_slogg$)", min_value=0.0, value=4.5, format="%.4f", help=r"Log base 10 of stellar surface gravity ($log_{10}(cm/s^2)$).")
            snr = st.number_input(r"10. Signal-to-Noise Ratio ($koi\_model\_snr$)", min_value=0.0, value=150.0, format="%.2f", help="The strength and cleanliness of the signal. High SNR is key to confirmation.")
            ror = st.number_input(r"11. Planet/Star Radius Ratio ($koi\_ror$)", min_value=0.0, value=0.02, format="%.5f", help="Ratio between the planet's radius and the star's radius.")

        st.subheader("False Positive Indicators (Binary Flags)")
        st.markdown("Checking these flags indicates the signal is likely a **False Positive** (FP).")
        
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            fpflag_nt_raw = st.checkbox("12. Not a Transit (NT)", help="The signal is not periodic or transient.")
        with col4:
            fpflag_ss_raw = st.checkbox("13. Stellar Eclipse (SS)", help="The signal comes from an eclipsing binary or a stellar FP.")
        with col5:
            fpflag_co_raw = st.checkbox("14. Centroid Shift (CO)", help="The center of light shifts during transit (indicates nearby source, not KOI).")
        with col6:
            fpflag_ec_raw = st.checkbox("15. Contamination (EC)", help="The signal is contaminated or has characteristics unrelated to the transit.")
            
        submitted = st.form_submit_button("Classify Object", type="primary")
        
    if submitted and model is not None and scaler is not None and imputer is not None:
        
        fpflag_nt = 1 if fpflag_nt_raw else 0
        fpflag_ss = 1 if fpflag_ss_raw else 0
        fpflag_co = 1 if fpflag_co_raw else 0
        fpflag_ec = 1 if fpflag_ec_raw else 0
        
        input_data = pd.DataFrame({
            'koi_period': [period], 'koi_duration': [duration], 'koi_prad': [prad], 
            'koi_teq': [teq], 'koi_steff': [steff], 'koi_srad': [srad], 
            'koi_smass': [smass],
            'koi_fpflag_nt': [fpflag_nt], 'koi_fpflag_ss': [fpflag_ss], 
            'koi_fpflag_co': [fpflag_co], 'koi_fpflag_ec': [fpflag_ec], 
            'koi_depth': [depth], 'koi_slogg': [slogg],
            'koi_model_snr': [snr], 'koi_ror': [ror]
        }, columns=KEPLER_FEATURES)
        
        X_imputed_array = imputer.transform(input_data)
        X_imputed = pd.DataFrame(X_imputed_array, columns=KEPLER_FEATURES)
        
        X_scaled = scaler.transform(X_imputed)
        
        prediction_raw = model.predict(X_scaled)[0]
        prediction_text, prediction_label = map_prediction(prediction_raw)
        
        probabilities = model.predict_proba(X_scaled)[0]
        prob_fp = probabilities[0] * 100
        prob_cand = probabilities[1] * 100
        prob_conf = probabilities[2] * 100
        
        st.subheader("Classification Result")
        
        if prediction_raw == 2:
            st.success(f"**Predicted Classification:** {prediction_text}")
        elif prediction_raw == 1:
            st.warning(f"**Predicted Classification:** {prediction_text}")
        else:
            st.error(f"**Predicted Classification:** {prediction_text}")
        
        st.write("---")
        
        st.markdown("##### Class Probabilities")
        col_prob1, col_prob2, col_prob3 = st.columns(3)
        
        with col_prob1:
            st.metric("False Positive (0)", f"{prob_fp:.2f}%")
        with col_prob2:
            st.metric("Candidate (1)", f"{prob_cand:.2f}%")
        with col_prob3:
            st.metric("Confirmed (2)", f"{prob_conf:.2f}%")

        st.write("---")
        st.markdown("##### Model Confidence Visualization")

        prob_df = pd.DataFrame({
            'Class': ['False Positive', 'Candidate', 'Confirmed'],
            'Probability': [prob_fp, prob_cand, prob_conf]
        })

        color_map_single = {
            'False Positive': '#F44336', 
            'Candidate': '#FFC107', 
            'Confirmed': '#4CAF50'
        }

        chart_single = alt.Chart(prob_df).mark_bar().encode(
            x=alt.X('Probability', title='Probability (%)'),
            y=alt.Y('Class', sort=['False Positive', 'Candidate', 'Confirmed'], title=None),
            color=alt.Color(
                'Class', 
                scale=alt.Scale(domain=list(color_map_single.keys()), range=list(color_map_single.values())),
                legend=None
            ),
            tooltip=['Class', alt.Tooltip('Probability', format='.2f')]
        ).properties(
            title='Confidence Levels by Predicted Class'
        )

        rule = alt.Chart(pd.DataFrame({'Probability': [50]})).mark_rule(color='gray', strokeDash=[3, 3]).encode(
            x='Probability'
        )

        st.altair_chart(chart_single + rule, use_container_width=True)

