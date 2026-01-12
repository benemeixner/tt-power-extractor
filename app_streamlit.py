# app_streamlit.py
import streamlit as st
import pandas as pd

from analysis_core import read_tt_file, extract_last_minutes, to_1hz

st.set_page_config(page_title="TT Power Extractor", layout="wide")
st.title("Cycling TT: extract last 3 / 5 / 12 minutes of Power")

st.write(
    "Upload a TT CSV export. The app extracts the **last N minutes** of Power (handy when the file includes ~1 min unloaded pedalling at the start).\n"
    "If your file includes a cooldown *after* the TT, trim it to end at the TT finish before uploading."
)

uploaded = st.file_uploader("Upload CSV", type=["csv", "txt"])

c1, c2 = st.columns([1, 2])
with c1:
    minutes = st.selectbox("Segment length (minutes)", [3, 5, 12], index=0)
with c2:
    view_mode = st.radio("Resolution", ["1 Hz (per-second mean)", "Raw (as recorded)"], index=0, horizontal=True)

if uploaded is not None:
    try:
        df = read_tt_file(uploaded.getvalue())
        seg, summ = extract_last_minutes(df, minutes)

        view = to_1hz(seg) if view_mode.startswith("1 Hz") else seg[["t_rel_s", "Power"]].copy()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Mean Power (W)", f"{summ.mean_power_w:.1f}")
        m2.metric("Median (W)", f"{summ.median_power_w:.1f}")
        m3.metric("Max (W)", f"{summ.max_power_w:.0f}")
        m4.metric("Min (W)", f"{summ.min_power_w:.0f}")
        m5.metric("Seconds extracted", f"{summ.seconds_available:.1f}")

        st.subheader("Power trace")
        plot_df = view.rename(columns={"t_rel_s": "Time (s)"})
        st.line_chart(plot_df.set_index("Time (s)"))

        st.subheader("Data")
        st.dataframe(plot_df, use_container_width=True, height=360)

        csv_bytes = plot_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download extracted segment CSV",
            data=csv_bytes,
            file_name=f"tt_last_{minutes}min_power.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Could not parse/analyze the file: {e}")
        st.stop()

st.divider()
st.markdown("""### Run locally
```bash
pip install streamlit pandas numpy
streamlit run app_streamlit.py
```
""")
