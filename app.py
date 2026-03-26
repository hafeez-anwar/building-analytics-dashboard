#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="Building Sensor Dashboard", layout="wide", page_icon="🏢")

# ==========================================
# 2. Helper Functions (Parsing & Cleaning)
# ==========================================
def parse_filename(filename):
    match = re.search(r'(BG\d{3})(\d{2})\s+([A-Za-z]{3})', filename)
    if match:
        building, floor, area_code = match.group(1), match.group(2), match.group(3).upper()
    else:
        building, floor, area_code = "Unknown", "Unknown", "Unknown"
        for code in ['COL', 'COR', 'MEL', 'LIL']:
            if code in filename.upper():
                area_code = code
                break
                
    area_map = {
        'COL': 'Left Corridor', 'COR': 'Right Corridor',
        'MEL': 'Main Entrance Lobby', 'LIL': 'Lift Lobby'
    }
    area_name = area_map.get(area_code, 'Unknown Area')
    return building, floor, area_code, area_name

@st.cache_data
def load_and_clean_data(file_obj, filename):
    df = pd.read_csv(file_obj)
    building, floor, area_code, area_name = parse_filename(filename)
    
    if area_code in ['COL', 'COR', 'MEL']:
        rename_dict = {
            'field1': 'TypeA_Temp', 'field2': 'TypeA_Hum',
            'field4': 'TypeB_Temp', 'field5': 'TypeB_Hum',
            'field6': 'TypeB_CO2_ppm', 'field7': 'TypeB_Lux'
        }
    elif area_code == 'LIL':
        rename_dict = {
            'field1': 'TypeB_Temp', 'field2': 'TypeB_Hum',
            'field3': 'TypeB_CO2_ppm', 'field4': 'TypeB_Lux'
        }
    else:
        rename_dict = {col: col for col in df.columns if 'field' in col}

    df = df.rename(columns=rename_dict)
    
    cols_to_drop = ['latitude', 'longitude', 'elevation', 'status', 'entry_id', 'field3', 'field8']
    for col in cols_to_drop:
        if col in df.columns and col not in rename_dict.values():
            df = df.drop(columns=[col])
            
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.sort_values('created_at').reset_index(drop=True)
    
    sensor_columns = [col for col in rename_dict.values() if col in df.columns]
    for col in sensor_columns:
        df[col] = df[col].interpolate(method='linear').bfill().ffill()
        
    df['hour'] = df['created_at'].dt.hour
    df['date'] = df['created_at'].dt.date
    df['month'] = df['created_at'].dt.month_name()
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = df['created_at'].dt.day_name()
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=days, ordered=True)
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).map({True: 'Weekend', False: 'Weekday'})
    
    return df, sensor_columns, building, floor, area_code, area_name

# ==========================================
# 3. Sidebar: Upload & Mode Selection
# ==========================================
st.sidebar.title("🏢 Smart Building Analytics")
uploaded_file = st.sidebar.file_uploader("Upload Sensor CSV", type=['csv'])

if uploaded_file is None:
    st.info("👈 Please upload a dataset (e.g., 'BG16300 COL.csv') to launch the dashboard.")
    st.stop()

# Load Original Cached Data
original_df, sensors, bldg, flr, area_code, area_name = load_and_clean_data(uploaded_file, uploaded_file.name)

st.sidebar.success("File Processed Successfully!")

analysis_mode = st.sidebar.radio("Navigation:", [
    "1. Overall Overview", 
    "2. Single Field Analysis", 
    "3. Pair Analysis",
    "4. HVAC Energy Efficiency 💡"
])

# ==========================================
# 4. Outlier Filtering (Dynamic UI)
# ==========================================
# We create a working copy of the dataframe to filter
plot_df = original_df.copy()

if analysis_mode in ["2. Single Field Analysis", "3. Pair Analysis"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Outlier Filtering")
    st.sidebar.caption("Manually adjust the sliders below to clip extreme values out of the plots.")
    
    # Determine which fields need sliders based on the mode
    if analysis_mode == "2. Single Field Analysis":
        target = st.sidebar.selectbox("Select Field to Analyze & Filter:", sensors)
        fields_to_filter = [target]
    elif analysis_mode == "3. Pair Analysis":
        field1 = st.sidebar.selectbox("Select Primary Field (Y1):", sensors, index=0)
        field2 = st.sidebar.selectbox("Select Secondary Field (Y2):", sensors, index=1 if len(sensors) > 1 else 0)
        fields_to_filter = [field1, field2] if field1 != field2 else [field1]

    # Create the dynamic sliders and apply the filters
    for field in fields_to_filter:
        min_val = float(original_df[field].min())
        max_val = float(original_df[field].max())
        
        # Streamlit slider for the range
        user_range = st.sidebar.slider(
            f"{field} Range", 
            min_value=min_val, 
            max_value=max_val, 
            value=(min_val, max_val), # Default is the full range
            step=(max_val - min_val) / 100.0
        )
        
        # Filter the working dataframe based on slider input
        plot_df = plot_df[(plot_df[field] >= user_range[0]) & (plot_df[field] <= user_range[1])]
        
    st.sidebar.info(f"Showing {len(plot_df)} of {len(original_df)} data points.")
    st.sidebar.markdown("---")

# ==========================================
# 5. Dashboard Views (Using `plot_df`)
# ==========================================

if analysis_mode == "1. Overall Overview":
    st.title(f"{bldg} - {area_name} ({flr}): Overview")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Data Summary")
        st.dataframe(original_df[sensors].describe().round(2), use_container_width=True)
    
    with col2:
        st.subheader("Interactive Correlation Matrix")
        corr_matrix = original_df[sensors].corr().round(2)
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)

elif analysis_mode == "2. Single Field Analysis":
    st.title(f"Interactive Analysis: {target}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Daily Average Trend")
        daily_avg = plot_df.groupby('date')[target].mean().reset_index()
        fig1 = px.line(daily_avg, x='date', y=target, markers=True, color_discrete_sequence=['teal'])
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.subheader("Day vs. Hour Heatmap")
        pivot_data = plot_df.groupby(['day_of_week', 'hour'], observed=False)[target].mean().reset_index()
        fig2 = px.density_heatmap(pivot_data, x='hour', y='day_of_week', z=target, histfunc='avg', color_continuous_scale='Viridis')
        fig2.update_layout(yaxis={'categoryorder':'array', 'categoryarray':['Sunday','Saturday','Friday','Thursday','Wednesday','Tuesday','Monday']})
        st.plotly_chart(fig2, use_container_width=True)
        
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Weekly Profile")
        fig3 = px.box(plot_df, x='day_of_week', y=target, color='day_of_week')
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        
    with col4:
        st.subheader("Weekday vs. Weekend Distribution")
        fig4 = px.histogram(plot_df, x=target, color='is_weekend', barmode='overlay', marginal='box')
        fig4.update_traces(opacity=0.7)
        st.plotly_chart(fig4, use_container_width=True)

elif analysis_mode == "3. Pair Analysis":
    st.title("Pair Analysis: Interactions & Relationships")
        
    if field1 == field2:
        st.warning("Please select two different fields in the sidebar.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"Scatter: {field1} vs {field2}")
            sample_df = plot_df.sample(n=min(5000, len(plot_df)), random_state=42)
            fig_scatter = px.scatter(sample_df, x=field1, y=field2, color='hour', opacity=0.6, color_continuous_scale='Twilight')
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col2:
            st.subheader("Statistical Relationship")
            corr_val = plot_df[field1].corr(plot_df[field2])
            st.metric(label="Pearson Correlation (Filtered Data)", value=f"{corr_val:.3f}")

        st.subheader("Dual-Axis Hourly Profile")
        hourly_stats = plot_df.groupby('hour')[[field1, field2]].mean().reset_index()
        
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dual.add_trace(go.Scatter(x=hourly_stats['hour'], y=hourly_stats[field1], name=field1, mode='lines+markers'), secondary_y=False)
        fig_dual.add_trace(go.Scatter(x=hourly_stats['hour'], y=hourly_stats[field2], name=field2, mode='lines+markers', line=dict(dash='dash')), secondary_y=True)
        
        fig_dual.update_layout(xaxis_title='Hour of the Day', hovermode='x unified')
        fig_dual.update_yaxes(title_text=field1, secondary_y=False)
        fig_dual.update_yaxes(title_text=field2, secondary_y=True)
        st.plotly_chart(fig_dual, use_container_width=True)

elif analysis_mode == "4. HVAC Energy Efficiency 💡":
    st.title("Energy Waste Detector")
    st.markdown("Identifies times when the area is likely **unoccupied** (Lights Off & Low CO2), but the **HVAC is still cooling**.")
    
    req_sensors = ['TypeB_Lux', 'TypeB_CO2_ppm', 'TypeB_Temp']
    missing = [s for s in req_sensors if s not in original_df.columns]
    
    if missing:
        st.error(f"Cannot run efficiency algorithm. Missing required sensors: {', '.join(missing)}")
    else:
        # We use original_df here so users don't accidentally filter out the waste events!
        calc_df = original_df.copy()
        calc_df['is_wasted_cooling'] = (calc_df['TypeB_Lux'] < 10) & (calc_df['TypeB_CO2_ppm'] < 600) & (calc_df['TypeB_Temp'] < 26.0)
        
        total_hours_in_data = len(calc_df) / 6 
        wasted_hours = len(calc_df[calc_df['is_wasted_cooling']]) / 6
        waste_percentage = (wasted_hours / total_hours_in_data) * 100 if total_hours_in_data > 0 else 0
        
        col1, col2 = st.columns([1, 2])
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = waste_percentage,
                title = {'text': "% of Time Wasted Cooling"},
                number = {'suffix': "%"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgreen"},
                        {'range': [10, 30], 'color': "gold"},
                        {'range': [30, 100], 'color': "salmon"}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.metric("Total Wasted Hours", f"{wasted_hours:.1f} hrs")
            
        with col2:
            st.subheader("Timeline of Cooling Waste")
            waste_df = calc_df[calc_df['is_wasted_cooling']]
            
            if len(waste_df) > 0:
                fig_timeline = px.scatter(waste_df, x='created_at', y='TypeB_Temp', 
                                          color_discrete_sequence=['red'], 
                                          labels={'created_at': 'Time', 'TypeB_Temp': 'Maintained Temp (°C)'})
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.success("No wasted cooling hours detected!")
                
        csv_waste = waste_df[['created_at', 'day_of_week', 'hour', 'TypeB_Lux', 'TypeB_CO2_ppm', 'TypeB_Temp']].to_csv(index=False)
        st.download_button(label="📥 Download Wasted Cooling Log", data=csv_waste, file_name=f"{bldg}_{area_code}_waste_log.csv", mime='text/csv')

