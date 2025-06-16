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

    # Tambahkan filter slider berdasarkan tahun
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
                model = ExponentialSmoothing(df_filtered['PNBP'], trend='add', seasonal=None, initialization_method="estimated")
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=3)
                forecast.index = pd.date_range(start=df_filtered.index.max() + pd.DateOffset(years=1), periods=3, freq='YS')

                df_pred = pd.concat([df_filtered['PNBP'], forecast])

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    hovertemplate='Des %{x|%Y}<br>Prediksi: %{y:.2s}',
                    x=df_filtered.index, y=df_filtered['PNBP'],
                    mode='lines+markers', name='Aktual',
                    line=dict(color='blue', width=2), marker=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    hovertemplate='Des %{x|%Y}<br>Prediksi: %{y:.2s}',
                    x=forecast.index, y=forecast.values,
                    mode='lines+markers', name='Prediksi',
                    line=dict(color='orange', width=2, dash='dot'), marker=dict(color='orange')
                ))

                fig.update_layout(
                    margin=dict(t=40),
                    title=f'Prediksi BHP PNBP - {jenis}',
                    xaxis_title='Tahun',
                    yaxis_title='BHP_PNBP',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black', size=12),
                    xaxis=dict(
                        tickmode='array',
                        tickvals=df_pred.index,
                        tickformat='%Y',
                        title_font=dict(color='black'),
                        tickfont=dict(color='black'),
                        showgrid=True,
                        gridcolor='lightgray',
                        gridwidth=0.5
                    ),
                    yaxis=dict(
                        title_font=dict(color='black'),
                        tickfont=dict(color='black'),
                        showgrid=True,
                        gridcolor='lightgray',
                        gridwidth=0.5
                    ),
                    legend=dict(font=dict(color='black')),
                    hoverlabel=dict(bgcolor='white', font=dict(color='black')),
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

                mape = mean_absolute_percentage_error(df_filtered['PNBP'], model_fit.fittedvalues)
                mae = mean_absolute_error(df_filtered['PNBP'], model_fit.fittedvalues)
                rmse = np.sqrt(mean_squared_error(df_filtered['PNBP'], model_fit.fittedvalues))
                st.write(
                    "Performa model dievaluasi menggunakan tiga metrik utama untuk menilai akurasi terhadap data historis, "
                    "yaitu **MAPE** (Mean Absolute Percentage Error), **MAE** (Mean Absolute Error), dan **RMSE** (Root Mean Squared Error). "
                    "Hasil evaluasi sebagai berikut:"
                )
                st.write(f"📊 MAPE: {mape:.2%}")
                st.write(f"📊 MAE: Rp {mae:,.0f}")
                st.write(f"📊 RMSE: Rp {rmse:,.0f}")

            except Exception as e:
                st.warning(f"Gagal memproses prediksi untuk '{jenis}'. Periksa data atau parameter model.")
                st.text(str(e))
