# pyright: reportMissingModuleSource=false

import os
from io import BytesIO
from typing import Any, Dict, Optional, Union

import requests
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

# Use environment variable for API URL to avoid requiring a secrets.toml file
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")
st.title("Energy Forecast Dashboard")

# File uploader or fallback to processed data in repo
uploaded = st.file_uploader("Upload processed data (sample.parquet)", type=["parquet"])
df: Optional[pd.DataFrame] = None
ParquetLike = Union[str, Path, BytesIO]


def try_read_parquet(path_or_buffer: ParquetLike) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path_or_buffer)
    except Exception as e:
        st.warning(f"Failed to read parquet: {e}")
        return None

if uploaded:
    # uploaded is a BytesIO-like Streamlit UploadedFile — check size
    if getattr(uploaded, "size", None) in (None, 0):
        st.error("Uploaded file is empty. Please upload a non-empty parquet file.")
        st.stop()
    df = try_read_parquet(uploaded)
    if df is None:
        st.error("Uploaded parquet could not be read. Try exporting from your preprocessing step or upload the CSV fallback file.")
        st.stop()
else:
    sample_parquet = Path("data/processed/sample.parquet")
    sample_csv = Path("data/processed/sample.csv")
    if sample_parquet.exists() and sample_parquet.stat().st_size > 0:
        df = try_read_parquet(sample_parquet)
        if df is None:
            st.warning("Found sample.parquet but failed to read it. Falling back to CSV if available.")
    if df is None and sample_csv.exists() and sample_csv.stat().st_size > 0:
        try:
            df = pd.read_csv(sample_csv, parse_dates=["timestamp"])
        except Exception as e:
            st.error(f"Failed to read CSV fallback: {e}")
            st.stop()
    if df is None:
        st.info("No valid sample.parquet found. Place a valid parquet or CSV at data/processed/ or upload one.")
        # offer to regenerate a parquet from CSV if possible
        if sample_csv.exists() and sample_csv.stat().st_size > 0:
            if st.button("Regenerate sample.parquet from sample.csv"):
                try:
                    df_csv = pd.read_csv(sample_csv, parse_dates=["timestamp"])
                    df_csv.to_parquet(sample_parquet)
                    st.success("Wrote data/processed/sample.parquet — please rerun the app or refresh to load it.")
                except Exception as e:
                    st.error(f"Failed to write parquet: {e}")
        st.stop()

st.subheader("Historical Power")
fig = px.line(df, x='timestamp', y='power', title='Power (kW)')
st.plotly_chart(fig, use_container_width=True)

horizon = st.slider('Forecast horizon (minutes ahead)', 1, 240, 60)

if st.button('Get Forecast'):
    recent_defaults = list(map(float, df['power'].tail(120).tolist()))
    payload: Dict[str, Any] = {'horizon': int(horizon), 'recent_window': recent_defaults}
    try:
        rec_resp = requests.get(f"{API_URL}/recent?n=120", timeout=15)
        if rec_resp.ok:
            rec_json = rec_resp.json()
            payload['recent_window'] = rec_json.get('recent') or payload['recent_window']
            if rec_json.get('timestamps'):
                payload['recent_timestamps'] = rec_json['timestamps']
        else:
            st.warning(f"/recent request failed: {rec_resp.text}")
    except Exception as e:
        st.warning(f"Failed to refresh recent data: {e}")
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        if resp.ok:
            data = resp.json()
            st.write('Model predictions (ensemble):')
            st.json(data)
            st.success(f"Ensemble forecast: {data.get('ensemble'):.3f} kW")
        else:
            st.error(f"API error: {resp.text}")
    except Exception as e:
        st.error('API request failed: ' + str(e))


st.markdown("---")
st.subheader("Live (auto-refreshing) view")

# Embedded HTML/JS to poll backend /recent and /predict endpoints and render live Plotly chart + latest prediction.
from streamlit.components.v1 import html as st_html

_api = API_URL.rstrip('/')
html_code = """
<!doctype html>
<html>
    <head>
        <meta charset="utf-8" />
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div id="plot" style="width:100%;height:420px;"></div>
        <div id="pred" style="font-family: monospace; margin-top: 8px;"></div>
        <script>
            const apiBase = '{_api}';
                    async function fetchRecent(){
                        try{
                            const r = await fetch(apiBase + '/recent?n=120');
                            if(!r.ok) throw new Error('recent fetch ' + r.status);
                            const j = await r.json();
                            // prefer timestamps+recent, fall back to recent array
                            if(j.timestamps && j.recent){ return {timestamps: j.timestamps, recent: j.recent}; }
                            return {timestamps: null, recent: (j.recent || [])};
                        }catch(e){ console.error(e); return {timestamps: null, recent: []}; }
                    }
                    async function fetchPred(recent, timestamps){
                try{
                            const payload = { horizon: 60, recent_window: recent, recent_timestamps: timestamps };
                    const r = await fetch(apiBase + '/predict', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
                    if(!r.ok) { const t = await r.text(); throw new Error('predict ' + r.status + ' ' + t); }
                    return await r.json();
                }catch(e){ console.error(e); return null; }
            }
            let plotDiv = document.getElementById('plot');
            function draw(initialX, initialY){
                const trace = { x: initialX, y: initialY, mode: 'lines+markers', name: 'power' };
                const layout = { title: 'Live power (kW)', xaxis: {title: 'index'}, yaxis: {title: 'kW'} };
                Plotly.newPlot(plotDiv, [trace], layout, {responsive:true});
            }
                    async function tick(){
                        const rec = await fetchRecent();
                        const recent = rec.recent || [];
                        const timestamps = rec.timestamps || null;
                        if(recent.length>0){
                            const x = timestamps ? timestamps : Array.from({length: recent.length}, (_,i)=>i-recent.length+1);
                            Plotly.react(plotDiv, [{ x: x, y: recent, mode: 'lines+markers', name: 'power' }], {xaxis:{type: timestamps? 'date':'linear'}});
                              const pred = await fetchPred(recent, timestamps);
                            if(pred){
                                document.getElementById('pred').innerText = 'Latest predictions: ' + JSON.stringify(pred);
                            }
                        }
                    }
            // initial draw
            draw([], []);
            // poll every 5 seconds
            tick(); setInterval(tick, 5000);
        </script>
    </body>
</html>
"""
html_code = html_code.replace('{_api}', _api)

st_html(html_code, height=520, scrolling=True)
