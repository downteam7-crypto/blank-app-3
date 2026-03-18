import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- [1. Configuration & Data Loading] ---
st.set_page_config(page_title="Air Semantic Anchor", layout="wide")

import os

@st.cache_data
def load_data():
    # Load Jan 2025 Air Quality Data
    # Use relative path to the current script
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "202501-air.csv")
    df = pd.read_csv(file_path)
    # Convert '측정일시' to datetime (format: YYYYMMDDHH)
    # 2025010101 -> 2025-01-01 01:00
    df['측정일시'] = pd.to_datetime(df['측정일시'].astype(str), format='%Y%m%d%H', errors='coerce')
    # Drop rows with invalid dates
    df = df.dropna(subset=['측정일시'])
    # Clean missing values for PM10 and PM25 (replace with median for now)
    df['PM10'] = pd.to_numeric(df['PM10'], errors='coerce').fillna(df['PM10'].median())
    df['PM25'] = pd.to_numeric(df['PM25'], errors='coerce').fillna(df['PM25'].median())
    return df

try:
    df_air = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- [2. Semantic Anchor DB for Air Quality] ---
# Defines thresholds for PM10 (main metric for this demo)
AIR_ANCHOR_DB = {
    "쾌적": {
        "threshold": 30.0, "intensity": 0.2, "variance": 0.5, "freeze": True,
        "rephrase_over": ["보통", "나쁨"], 
        "rephrase_under": ["매우 좋음", "청정"], 
        "rephrase_aligned": ["맑음", "좋음"],
        "coord_x": -0.8, "coord_y": 0.8
    },
    "나쁨": {
        "threshold": 80.0, "intensity": 0.7, "variance": 1.5, "freeze": False,
        "rephrase_over": ["보통", "다소 높음"], 
        "rephrase_under": ["매우 나쁨", "최악"], 
        "rephrase_aligned": ["주의 필요", "먼지 많음"],
        "coord_x": 0.5, "coord_y": -0.5
    },
    "숨 막히는": {
        "threshold": 150.0, "intensity": 0.95, "variance": 3.0, "freeze": False,
        "rephrase_over": ["나쁨", "매우 나쁨"], 
        "rephrase_under": ["재난 수준", "통제 불각"], 
        "rephrase_aligned": ["심각한 오염", "주의보 발령"],
        "coord_x": 0.9, "coord_y": -0.9
    },
    "안정세": {
        "threshold": 10.0, "intensity": 0.1, "variance": 0.2, "freeze": True, # This refers to 'change' (variance)
        "rephrase_over": ["변동성 확대", "불안정"], 
        "rephrase_under": ["정지 상태", "완전 고착"], 
        "rephrase_aligned": ["보합", "일정한 수준"],
        "coord_x": 0.0, "coord_y": 0.0
    }
}

def analyze_air_semantics(expression, observed_pm10):
    if expression not in AIR_ANCHOR_DB: return None
    anchor = AIR_ANCHOR_DB[expression]
    
    # Logic similar to reference: distance from threshold
    distance_ratio = abs(observed_pm10 - anchor["threshold"]) / (anchor["threshold"] + 1e-5)
    confidence = 1 / (1 + anchor["variance"])
    alignment_score = np.exp(-distance_ratio) * confidence
    
    observed_intensity = (observed_pm10 / (anchor["threshold"] + 1e-5)) * anchor["intensity"]
    intensity_gap = anchor["intensity"] - observed_intensity
    distortion_magnitude = min(1.0, abs(intensity_gap))
    
    if intensity_gap > 0.15:
        distortion_type = "과장 (Overstatement)"
        gauge_color = "darkred"
        rephrase_list = anchor["rephrase_over"]
        context_msg = f"실제 수치({observed_pm10}㎍/㎥)보다 단어의 강도가 강합니다. 과장된 표현일 수 있습니다."
    elif intensity_gap < -0.15:
        distortion_type = "축소/은폐 (Understatement)"
        gauge_color = "darkblue"
        rephrase_list = anchor["rephrase_under"]
        context_msg = f"실제 수치({observed_pm10}㎍/㎥)가 단어의 기준을 훨씬 초과했습니다. 더 강한 표현이 필요할 수 있습니다."
    else:
        distortion_type = "정합 (Aligned)"
        gauge_color = "green"
        rephrase_list = anchor["rephrase_aligned"]
        context_msg = "현재 공기 상태와 단어의 의미가 잘 맞아떨어집니다."

    return {
        "expression": expression,
        "observed_val": observed_pm10,
        "anchor_threshold": anchor['threshold'],
        "alignment_score": alignment_score,
        "distortion_magnitude": distortion_magnitude,
        "distortion_type": distortion_type,
        "gauge_color": gauge_color,
        "confidence_score": confidence,
        "rephrase": ", ".join(rephrase_list),
        "context_msg": context_msg,
        "coord_x": anchor["coord_x"],
        "coord_y": anchor["coord_y"]
    }

# --- [3. UI Implementation] ---
st.title("🌬️ Air Quality Semantic Anchor")
st.write("실제 대기질 데이터(2025.01)와 언어 표현의 정합성을 분석하는 시스템입니다.")

# Sidebar Selection
with st.sidebar:
    st.header("Search Filters")
    all_regions = sorted(df_air['지역'].unique())
    selected_region = st.selectbox("Select Region", all_regions, index=all_regions.index("서울 중구") if "서울 중구" in all_regions else 0)
    
    region_stations = sorted(df_air[df_air['지역'] == selected_region]['측정소명'].unique())
    selected_station = st.selectbox("Select Station", region_stations)
    
    st.divider()
    st.subheader("Select Date & Time")
    # Will be populated after filtering data below
    station_times_placeholder = st.empty()
    
    st.divider()
    st.caption("Data Source: 한국환경공단 (Jan 2025)")

# Filter data for the region/station
df_filtered = df_air[(df_air['지역'] == selected_region) & (df_air['측정소명'] == selected_station)].sort_values('측정일시')

if df_filtered.empty:
    st.warning("No data found for the selected station.")
    st.stop()

# Build the timestamp selector in the sidebar placeholder
available_times = df_filtered['측정일시'].dt.strftime('%Y-%m-%d %H:%M').tolist()
with station_times_placeholder:
    selected_time_str = st.selectbox("Measurement Time", available_times, index=len(available_times)-1)

# Input Section
st.markdown("### Step 1. 기사/평가 문구 입력")
col1, col2, col3 = st.columns(3)
if col1.button("Sample: 오늘 공기 정말 쾌적하다 (과장?)"):
    st.session_state['air_input'] = "오늘 공기는 정말 쾌적해서 산책하기 딱 좋습니다."
if col2.button("Sample: 대기질이 나쁨 수준입니다 (정합?)"):
    st.session_state['air_input'] = "수도권 대기질이 나쁨 수준으로 확인되어 주의가 필요합니다."
if col3.button("Sample: 숨 막히는 먼지 (데이터 확인)"):
    st.session_state['air_input'] = "미세먼지 농도가 치솟아 숨 막히는 하루입니다."

air_text = st.text_area("Input air quality description", value=st.session_state.get('air_input', ''), height=100)
analyze_btn = st.button("Analyze Alignment", type="primary")

if analyze_btn and air_text:
    # Use the user-selected time record
    selected_record = df_filtered[df_filtered['측정일시'].dt.strftime('%Y-%m-%d %H:%M') == selected_time_str]
    latest_record = selected_record.iloc[0] if not selected_record.empty else df_filtered.iloc[-1]
    observed_pm10 = latest_record['PM10']
    
    # Detect keyword
    detected_expr = None
    if "숨 막" in air_text or "심각" in air_text: detected_expr = "숨 막히는"
    elif "나쁨" in air_text or "먼지" in air_text: detected_expr = "나쁨"
    elif "쾌적" in air_text or "좋음" in air_text: detected_expr = "쾌적"
    elif "안정" in air_text or "보합" in air_text: detected_expr = "안정세"
    
    if detected_expr:
        res = analyze_air_semantics(detected_expr, observed_pm10)
        
        st.divider()
        date_str = latest_record['측정일시'].strftime('%Y-%m-%d %H:%M') if pd.notnull(latest_record['측정일시']) else "Unknown Date"
        st.markdown(f"### Results for {selected_region} ({date_str})")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Detected Concept", res['expression'])
            st.write(f"- 실측 PM10: **{res['observed_val']}㎍/㎥**")
            st.write(f"- Anchor 기준: **{res['anchor_threshold']}㎍/㎥**")
            
        with c2:
            st.markdown(f"**왜곡도: <span style='color:{res['gauge_color']}'>{res['distortion_type']}</span>**", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res['distortion_magnitude'],
                gauge={'axis': {'range': [0, 1]}, 'bar': {'color': res['gauge_color']}}
            ))
            fig_gauge.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with c3:
            st.success(f"**추천 표현:** {res['rephrase']}")
            st.info(f"**해석:** {res['context_msg']}")

    else:
        st.warning("분석 가능한 키워드('쾌적', '나쁨', '숨 막히는', '안정')가 발견되지 않았습니다.")

st.divider()

# --- [4. Dashboards & Charts] ---
st.markdown("### Step 2. Data Visualizations")
tab1, tab2, tab3 = st.tabs(["📉 Month Trend", "🗺️ Semantic Axis", "📋 Raw Data"])

with tab1:
    st.subheader(f"2025년 01월 {selected_station} 대기질 추이")
    fig_line = px.line(df_filtered, x='측정일시', y=['PM10', 'PM25'], 
                      title=f"{selected_station} 미세먼지 농도 추이",
                      labels={'value': '농도 (㎍/㎥)', 'variable': '항목'})
    fig_line.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="나쁨 기준(80)")
    fig_line.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="매우 나쁨(150)")
    st.plotly_chart(fig_line, use_container_width=True)

with tab2:
    st.subheader("Linguistic Semantic Map")
    df_map = pd.DataFrame([
        {"Word": k, "X": v["coord_x"], "Y": v["coord_y"], "Threshold": v["threshold"]} 
        for k, v in AIR_ANCHOR_DB.items()
    ])
    fig_map = px.scatter(df_map, x="X", y="Y", text="Word", size="Threshold",
                        labels={"X": "쾌적 ↔ 오염", "Y": "조용 ↔ 역동"},
                        title="대기질 표현의 심리 좌표계")
    fig_map.update_traces(textposition='top center')
    fig_map.update_layout(height=500)
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.dataframe(df_filtered, use_container_width=True)
