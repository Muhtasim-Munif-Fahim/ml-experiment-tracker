"""Streamlit dashboard for ML Experiment Tracker"""

from __future__ import annotations

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Dict, Any
import requests

API_BASE = "http://localhost:8000"


def api_get(endpoint: str) -> dict:
    try:
        resp = requests.get(f"http://localhost:8000{endpoint}")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(endpoint: str, data: dict) -> dict:
    try:
        resp = requests.post(f"http://localhost:8000{endpoint}", json=data)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


st.set_page_config(page_title="ML Experiment Tracker", page_icon="📊", layout="wide")

st.title("🧪 ML Experiment Tracker")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Page", ["Experiments", "Runs", "Compare", "Artifacts"])

if st.sidebar.button("Refresh"):
    st.rerun()

# Experiments page
if page == "Experiments":
    st.header("Experiments")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Experiments")
    with col2:
        if st.button("➕ New Experiment"):
            st.session_state["show_create_exp"] = True

    if st.session_state.get("show_create_exp"):
        with st.form("create_exp"):
            name = st.text_input("Experiment Name")
            desc = st.text_area("Description")
            tags = st.text_input("Tags (comma-separated)")
            if st.form_submit_button("Create"):
                resp = requests.post("http://localhost:8000/experiments/", json={"name": name, "description": desc, "tags": [t.strip() for t in tags.split(",") if tags]})
                if resp.status_code == 200:
                    st.success("Experiment created!")
                    st.rerun()

    experiments = api_get("/experiments/")
    if experiments:
        for exp in experiments:
            with st.expander(f"{exp['name']} ({exp['id'][:8]})"):
                st.write(exp.get("description", ""))
                st.caption(f"Created: {exp['created_at'][:19]} | ID: {exp['id'][:8]}")
                col1, col2, col3 = st.columns(3)
                if st.button("View Runs", key=f"view_{exp['id']}"):
                    st.session_state["selected_exp"] = exp["id"]
                    st.rerun()
                if st.button("New Run", key=f"run_{exp['id']}"):
                    st.session_state["new_run_exp"] = exp["id"]
                    st.rerun()
                if st.button("Delete", key=f"del_{exp['id']}"):
                    if st.confirm("Delete experiment?"):
                        requests.delete(f"http://localhost:8000/experiments/{exp['id']}")
                        st.rerun()

        if st.session_state.get("selected_exp"):
            exp_id = st.session_state["selected_exp"]
            st.subheader(f"Runs for {exp_id[:8]}")
            runs = api_get(f"/experiments/{exp_id}/runs/")
            if runs:
                df = pd.DataFrame(runs)
                st.dataframe(df[["id", "name", "status", "created_at"]])
                if st.button("New Run", key=f"new_run_{exp_id}"):
                    st.session_state["new_run_exp"] = exp_id
                    st.rerun()

# Runs page
elif page == "Runs":
    st.header("Runs")
    exp_id = st.session_state.get("selected_exp")
    if not exp_id:
        st.warning("Select an experiment first")
    else:
        runs = api_get(f"/experiments/{exp_id}/runs/")
        if runs:
            df = pd.DataFrame(runs)
            st.dataframe(df[["id", "name", "status", "created_at", "finished_at"]])
            selected = st.selectbox("Select run", runs, format_func=lambda x: f"{x['name']} ({x['id'][:8]})")
            if selected:
                run = api_get(f"/runs/{selected}")
                st.json(run)

# Artifacts page
elif page == "Artifacts":
    st.header("Artifacts")
    run_id = st.session_state.get("selected_run")
    if run_id:
        artifacts = api_get(f"/runs/{run_id}/artifacts/")
        if artifacts:
            for art in artifacts:
                st.write(f"**{art['name']}** ({art['type']}) - {art['size_bytes']} bytes")
                if art['type'] == 'model':
                    st.download_button("Download", data=requests.get(f"http://localhost:8000/runs/{run_id}/artifacts/{art['name']}").content, file_name=art['name'])

# Compare page
elif page == "Compare":
    st.header("Compare Runs")
    exp_id = st.session_state.get("selected_exp")
    if exp_id:
        runs = api_get(f"/experiments/{exp_id}/runs/")
        if runs:
            selected = st.multiselect("Select runs to compare", runs, format_func=lambda x: f"{x['name']} ({x['id'][:8]})")
            if len(selected) >= 2:
                dfs = []
                for run in selected:
                    run_data = api_get(f"/runs/{run['id']}")
                    metrics = pd.DataFrame(run_data.get("metrics", []))
                    if not metrics.empty:
                        st.line_chart(pd.DataFrame(run["metrics"]).set_index("step")["value"])
                        st.line_chart(pd.DataFrame({run['name']: run['metrics']['value'] for run in selected}))

st.sidebar.markdown("---")
st.sidebar.caption("ML Experiment Tracker v0.1")