
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import warnings

# Hapus peringatan konvergensi
warnings.filterwarnings("ignore", message=".*Optimization failed to converge.*")

st.set_page_config(page_title="Prediksi PNBP", layout="centered", page_icon="📈")

# CSS styling
st.markdown(
    """
    <style>
    [data-testid="stTabs"] button {
        color: black !important;
        font-weight: normal;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: red !important;
        font-weight: bold !important;
    }

    body {
        color: black !important;
        background-color: white !important;
    }
    .reportview-container, .main, .block-container {
        color: black !important;
        background-color: white !important;
    }
    .stPlotlyChart {
        border: 1px solid #ccc;
        border-radius: 8px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        padding: 10px;
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("📈 Model Prediksi BHP PNBP Ditjen Infradig Berbasis Data Science")

uploaded_file = st.file_uploader("Upload file Excel Dataset PNBP", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name='Tabel_Target_dan_Realisasi_PNBP')

    df_long = df.melt(id_vars='Jenis PNBP', var_name='Tahun', value_name='PNBP')
    df_long['Tahun'] = pd.to_datetime(df_long['Tahun'], format='%Y')
    df_long['PNBP'] = df_long['PNBP'].astype(str).str.replace(".", "", regex=False).astype(float)

    min_year = df_long['Tahun'].dt.year.min()
    max_year = df_long['Tahun'].dt.year.max()
    tahun_range = st.slider(
        "Pilih rentang tahun untuk ditampilkan:",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(min_year), int(max_year)),
        step=1
    )
    df_long = df_long[df_long['Tahun'].dt.year.between(tahun_range[0], tahun_range[1])]

    jenis_list = df_long['Jenis PNBP'].unique().tolist()
    tabs = st.tabs([f"{jenis}" for jenis in jenis_list])

    for i, jenis in enumerate(jenis_list):
        with tabs[i]:
            st.subheader(f"📈 {jenis}")
            df_filtered = df_long[df_long['Jenis PNBP'] == jenis].copy()
            df_filtered = df_filtered.set_index('Tahun').asfreq('YS')

            try:
                model = ExponentialSmoothing(df_filtered['PNBP'], trend='add', seasonal='add', seasonal_periods=1, initialization_method="estimated")
                model_fit = model.fit()

                fitted = model_fit.fittedvalues
                fitted.index = df_filtered.index

                forecast = model_fit.forecast(steps=3)
                forecast.index = pd.date_range(start=df_filtered.index.max() + pd.DateOffset(years=1), periods=3, freq='YS')

                df_pred_all = pd.concat([fitted, forecast])
                df_actual = df_filtered['PNBP']

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_actual.index, y=df_actual, mode='lines+markers', name='Aktual',
                                         line=dict(color='blue'), marker=dict(color='blue')))
                fig.add_trace(go.Scatter(x=df_pred_all.index, y=df_pred_all, mode='lines+markers', name='Prediksi',
                                         line=dict(color='orange', dash='dot'), marker=dict(color='orange')))
                fig.update_layout(
                    title=f'Prediksi BHP PNBP 2014–2027 - {jenis}',
                    xaxis_title='Tahun',
                    yaxis_title='PNBP',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📅 Prediksi 3 Tahun ke Depan")
                forecast_df = pd.DataFrame({
                    'Tahun': forecast.index.year,
                    'Prediksi PNBP': forecast.values
                })
                forecast_df['Prediksi PNBP'] = forecast_df['Prediksi PNBP'].apply(lambda x: f"Rp {x:,.0f}")
                for index, row in forecast_df.iterrows():
                    st.markdown(f"**{row['Tahun']}** : {row['Prediksi PNBP']}")

                mape = mean_absolute_percentage_error(df_actual, model_fit.fittedvalues)
                mae = mean_absolute_error(df_actual, model_fit.fittedvalues)
                rmse = np.sqrt(mean_squared_error(df_actual, model_fit.fittedvalues))
                st.write("📊 **Evaluasi Performa (2014–2024)**")
                st.write(f"MAPE: {mape:.2%}")
                st.write(f"MAE: Rp {mae:,.0f}")
                st.write(f"RMSE: Rp {rmse:,.0f}")

            except Exception as e:
                st.warning(f"Gagal memproses prediksi untuk '{jenis}'. Periksa data atau parameter model.")
                st.text(str(e))
