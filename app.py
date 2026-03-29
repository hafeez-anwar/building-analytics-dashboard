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
            'field6': 'TypeB_CO2_ppm', 'field7': 'TypeB_Luminance'
        }
    elif area_code == 'LIL':
        rename_dict = {
            'field1': 'TypeB_Temp', 'field2': 'TypeB_Hum',
            'field3': 'TypeB_CO2_ppm', 'field4': 'TypeB_Luminance'
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

original_df, sensors, bldg, flr, area_code, area_name = load_and_clean_data(uploaded_file, uploaded_file.name)

st.sidebar.success("File Processed Successfully!")

analysis_mode = st.sidebar.radio("Navigation:", [
    "1. Overall Overview", 
    "2. Single Field Analysis", 
    "3. Pair Analysis",
    "4. HVAC Energy Efficiency 💡",
    "5. Smart Alerts & Diagnostics 🚨"
])

# ==========================================
# 4. Outlier Filtering (Dynamic UI)
# ==========================================
plot_df = original_df.copy()

if analysis_mode in ["2. Single Field Analysis", "3. Pair Analysis"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Outlier Filtering")
    
    if analysis_mode == "2. Single Field Analysis":
        target = st.sidebar.selectbox("Select Field to Analyze & Filter:", sensors)
        fields_to_filter = [target]
    elif analysis_mode == "3. Pair Analysis":
        field1 = st.sidebar.selectbox("Select Primary Field (Y1):", sensors, index=0)
        field2 = st.sidebar.selectbox("Select Secondary Field (Y2):", sensors, index=1 if len(sensors) > 1 else 0)
        fields_to_filter = [field1, field2] if field1 != field2 else [field1]

    for field in fields_to_filter:
        min_val, max_val = float(original_df[field].min()), float(original_df[field].max())
        user_range = st.sidebar.slider(f"{field} Range", min_value=min_val, max_value=max_val, value=(min_val, max_val), step=(max_val - min_val) / 100.0)
        plot_df = plot_df[(plot_df[field] >= user_range[0]) & (plot_df[field] <= user_range[1])]
        
    st.sidebar.info(f"Showing {len(plot_df)} of {len(original_df)} data points.")

# ==========================================
# 5. Dashboard Views
# ==========================================

if analysis_mode == "1. Overall Overview":
    st.title(f"{bldg} - {area_name} ({flr}): Overview")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Data Summary")
        st.dataframe(original_df[sensors].describe().round(2), use_container_width=True)
    with col2:
        st.subheader("Interactive Correlation Matrix")
        fig = px.imshow(original_df[sensors].corr().round(2), text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)

elif analysis_mode == "2. Single Field Analysis":
    st.title(f"Interactive Analysis: {target}")
    col1, col2 = st.columns(2)
    with col1:
        daily_avg = plot_df.groupby('date')[target].mean().reset_index()
        fig1 = px.line(daily_avg, x='date', y=target, markers=True, title="Daily Average Trend", color_discrete_sequence=['teal'])
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        pivot_data = plot_df.groupby(['day_of_week', 'hour'], observed=False)[target].mean().reset_index()
        fig2 = px.density_heatmap(pivot_data, x='hour', y='day_of_week', z=target, histfunc='avg', title="Day vs. Hour Heatmap", color_continuous_scale='Viridis')
        fig2.update_layout(yaxis={'categoryorder':'array', 'categoryarray':['Sunday','Saturday','Friday','Thursday','Wednesday','Tuesday','Monday']})
        st.plotly_chart(fig2, use_container_width=True)
        
    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.box(plot_df, x='day_of_week', y=target, color='day_of_week', title="Weekly Profile")
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = px.histogram(plot_df, x=target, color='is_weekend', barmode='overlay', marginal='box', title="Weekday vs. Weekend Distribution")
        st.plotly_chart(fig4, use_container_width=True)

elif analysis_mode == "3. Pair Analysis":
    st.title("Pair Analysis: Interactions & Relationships")
    if field1 == field2:
        st.warning("Please select two different fields in the sidebar.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            sample_df = plot_df.sample(n=min(5000, len(plot_df)), random_state=42)
            fig_scatter = px.scatter(sample_df, x=field1, y=field2, color='hour', opacity=0.6, title=f"Scatter: {field1} vs {field2}", color_continuous_scale='Twilight')
            st.plotly_chart(fig_scatter, use_container_width=True)
        with col2:
            st.subheader("Statistical Relationship")
            corr_val = plot_df[field1].corr(plot_df[field2])
            st.metric(label="Pearson Correlation (Filtered)", value=f"{corr_val:.3f}")

        hourly_stats = plot_df.groupby('hour')[[field1, field2]].mean().reset_index()
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dual.add_trace(go.Scatter(x=hourly_stats['hour'], y=hourly_stats[field1], name=field1, mode='lines+markers'), secondary_y=False)
        fig_dual.add_trace(go.Scatter(x=hourly_stats['hour'], y=hourly_stats[field2], name=field2, mode='lines+markers', line=dict(dash='dash')), secondary_y=True)
        fig_dual.update_layout(title="Dual-Axis Hourly Profile", xaxis_title='Hour of the Day', hovermode='x unified')
        st.plotly_chart(fig_dual, use_container_width=True)

elif analysis_mode == "4. HVAC Energy Efficiency 💡":
    st.title("HVAC Energy Waste Detector")
    st.markdown("Identifies times when the area is likely unoccupied, but the HVAC is still actively cooling.")
    
    with st.expander("⚙️ View Algorithm Logic & Thresholds"):
        st.markdown("""
        **How do we define 'Wasted Cooling'?** We use a **2-Hour Thermal Lag Buffer**. The system groups contiguous blocks of time where the room is:
        * **`Luminance < 10` (Lights Off)**
        * **`CO2 < 600 ppm` (Unoccupied)**
        * **`Temperature < 26.0°C` (Actively Cooled)**
        
        If this state lasts for **more than 2 hours**, it is logged. The first 2 hours are considered "natural thermal lag" and are subtracted from the penalty.
        """)
    
    req_sensors = ['TypeB_Luminance', 'TypeB_CO2_ppm', 'TypeB_Temp']
    if not all(s in original_df.columns for s in req_sensors):
        st.error("Missing required sensors for this analysis.")
    else:
        calc_df = original_df.copy()
        
        # 1. Evaluate the base condition
        calc_df['base_waste_cond'] = (calc_df['TypeB_Luminance'] < 10) & (calc_df['TypeB_CO2_ppm'] < 600) & (calc_df['TypeB_Temp'] < 26.0)
        
        # 2. Group contiguous blocks
        calc_df['waste_group'] = (calc_df['base_waste_cond'] != calc_df['base_waste_cond'].shift()).cumsum()
        true_blocks = calc_df[calc_df['base_waste_cond']]
        
        wasted_hours_total = 0.0
        
        if len(true_blocks) > 0:
            # 3. Aggregate valid blocks
            event_summary = true_blocks.groupby('waste_group').agg(
                From_DT=('created_at', 'min'),
                To_DT=('created_at', 'max'),
                Avg_Temp=('TypeB_Temp', 'mean')
            )
            
            # 4. Calculate strict duration
            event_summary['Duration'] = event_summary['To_DT'] - event_summary['From_DT']
            event_summary['Total Duration (hrs)'] = (event_summary['Duration'].dt.total_seconds() / 3600).round(2)
            
            # 5. FILTER: Keep events >= 2 hours
            valid_events = event_summary[event_summary['Total Duration (hrs)'] >= 1.95].copy()
            
            if len(valid_events) > 0:
                # Calculate penalized waste
                valid_events['Wasted Hours (Penalty)'] = (valid_events['Total Duration (hrs)'] - 2.0).clip(lower=0).round(2)
                wasted_hours_total = valid_events['Wasted Hours (Penalty)'].sum()
                
                # Format for display table
                valid_events['Start Date'] = valid_events['From_DT'].dt.strftime('%Y-%m-%d')
                valid_events['Day'] = valid_events['From_DT'].dt.strftime('%A')
                valid_events['From Time'] = valid_events['From_DT'].dt.strftime('%H:%M')
                valid_events['End Date'] = valid_events['To_DT'].dt.strftime('%Y-%m-%d')
                valid_events['To Time'] = valid_events['To_DT'].dt.strftime('%H:%M')
                valid_events['Avg Temp (°C)'] = valid_events['Avg_Temp'].round(2)
                
                display_cols = ['Start Date', 'Day', 'From Time', 'End Date', 'To Time', 'Total Duration (hrs)', 'Wasted Hours (Penalty)', 'Avg Temp (°C)']
                display_waste = valid_events[display_cols].reset_index(drop=True)
                
                # --- PREPARE DATA FOR VISUALIZATIONS ---
                valid_group_ids = valid_events.index
                plot_df = calc_df[calc_df['waste_group'].isin(valid_group_ids) & calc_df['base_waste_cond']].copy()
                plot_df['created_at'] = pd.to_datetime(plot_df['created_at'])
                
                # Strictly penalized data (Drops first 12 rows / 2 hrs)
                plot_df = plot_df.sort_values(['waste_group', 'created_at'])
                plot_df_penalty = plot_df.groupby('waste_group').tail(-12).copy()
                plot_df_penalty['Day_of_Week'] = plot_df_penalty['created_at'].dt.day_name()
                plot_df_penalty['Hour'] = plot_df_penalty['created_at'].dt.hour
                
                # ==========================================
                # UI LAYOUT SECTION
                # ==========================================
                
                st.markdown("---")
                st.subheader("📊 Executive Summary")
                
                # ROW 1: Gauge and Bar Chart Side-by-Side
                col_gauge, col_bar = st.columns([1, 1.5])
                
                with col_gauge:
                    st.metric("Total Penalized Wasted Hours", f"{wasted_hours_total:.1f} hrs")
                    max_gauge_val = max(50, wasted_hours_total * 1.5) if wasted_hours_total > 0 else 50
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=wasted_hours_total,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [None, max_gauge_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "#EF553B"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, max_gauge_val * 0.3], 'color': "#e6f2ff"},
                                {'range': [max_gauge_val * 0.3, max_gauge_val * 0.7], 'color': "#cce5ff"}],
                        }
                    ))
                    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col_bar:
                    day_dist = plot_df_penalty.groupby('Day_of_Week', observed=False).size().reset_index(name='intervals')
                    day_dist['Penalty Hours'] = (day_dist['intervals'] / 6.0).round(1)
                    waste_by_day = day_dist.sort_values(by='Penalty Hours', ascending=True)
                    
                    if not waste_by_day.empty and waste_by_day['Penalty Hours'].sum() > 0:
                        fig_worst_day = px.bar(
                            waste_by_day, 
                            x='Penalty Hours', 
                            y='Day_of_Week', 
                            orientation='h', 
                            title="Worst Offenders (Penalized Hours by Day)",
                            text_auto='.1f',
                            color='Penalty Hours',
                            color_continuous_scale='Reds'
                        )
                        fig_worst_day.update_layout(height=320, showlegend=False, coloraxis_showscale=False, margin=dict(l=10, r=20, t=40, b=20))
                        st.plotly_chart(fig_worst_day, use_container_width=True)

                # ROW 2: Full Width Heatmap
                st.markdown("---")
                st.subheader("📅 Scheduling Flaw Heatmap")
                st.markdown("Pinpoints the exact days and times with the highest confirmed energy waste, helping you identify scheduling errors.")
                
                heatmap_data = plot_df_penalty.groupby(['Day_of_Week', 'Hour'], observed=False).size().reset_index(name='intervals')
                heatmap_data['Waste_Hours'] = heatmap_data['intervals'] / 6.0
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_pivot = heatmap_data.pivot(index='Day_of_Week', columns='Hour', values='Waste_Hours').reindex(days_order).fillna(0)
                
                for h in range(24):
                    if h not in heatmap_pivot.columns:
                        heatmap_pivot[h] = 0
                heatmap_pivot = heatmap_pivot[range(24)]
                
                fig_heat = px.imshow(
                    heatmap_pivot,
                    labels=dict(x="Hour of Day", y="Day of the Week", color="Hours of Waste"),
                    x=[f"{h:02d}:00" for h in range(24)],
                    y=days_order,
                    color_continuous_scale="Reds",
                    aspect="auto",
                    text_auto=".1f"
                )
                fig_heat.update_xaxes(side="top")
                fig_heat.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=350)
                st.plotly_chart(fig_heat, use_container_width=True)
                
                # ROW 3: Engineering Deep Dive (Scatter & Table)
                st.markdown("---")
                st.subheader("🛠️ Engineering Deep Dive")
                
                fig_timeline = px.scatter(
                    plot_df, x='created_at', y='TypeB_Temp', 
                    color_discrete_sequence=['red'], 
                    title="Temperature Profile During Waste Events"
                )
                fig_timeline.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                st.markdown("**Detailed Waste Event Log**")
                st.dataframe(display_waste, use_container_width=True, hide_index=True)
                
            else:
                st.success("No sustained wasted cooling detected! System is shutting down efficiently.")
        else:
            st.success("No sustained wasted cooling detected! System is shutting down efficiently.")

elif analysis_mode == "5. Smart Alerts & Diagnostics 🚨":
    st.title("Smart Alerts & Diagnostics")
    st.markdown("Automated fault detection algorithms scanning for operational inefficiencies and hardware failures.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["💡 Ghost Lighting", "💨 Ventilation Failure", "❄️ Freezer Zone", "🔧 Sensor Health"])
    calc_df = original_df.copy()
    
    # --- TAB 1: GHOST LIGHTING (UPGRADED ENGINE) ---
    with tab1:
        st.subheader("Ghost Lighting Detector")
        with st.expander("⚙️ View Algorithm Logic & Thresholds"):
            st.markdown("""
            **How do we define 'Ghost Lighting'?** We use a **30-Minute Grace Period**. The system groups contiguous blocks of time where the room is:
            * **`Luminance > 100` (Lights Actively On)**
            * **`CO2 < 450 ppm` (Absolute Baseline / Zero Human Presence)**
            
            If this state lasts for **more than 30 minutes**, it is logged. The first 30 minutes are considered a normal "stepped out for a break" buffer and are subtracted from the penalty.
            """)
        
        req_sensors = ['TypeB_Luminance', 'TypeB_CO2_ppm']
        if not all(s in calc_df.columns for s in req_sensors):
            st.error("Missing required sensors for this analysis (Luminance and CO2).")
        else:
            # Isolate data for this specific tab to prevent column overlap
            ghost_calc = calc_df.copy()
            
            # 1. Evaluate the base condition using your strict thresholds
            ghost_calc['base_waste_cond'] = (ghost_calc['TypeB_Luminance'] > 100) & (ghost_calc['TypeB_CO2_ppm'] < 450)
            
            # 2. Group contiguous blocks
            ghost_calc['waste_group'] = (ghost_calc['base_waste_cond'] != ghost_calc['base_waste_cond'].shift()).cumsum()
            true_blocks = ghost_calc[ghost_calc['base_waste_cond']]
            
            wasted_hours_total = 0.0
            
            if len(true_blocks) > 0:
                # 3. Aggregate valid blocks
                event_summary = true_blocks.groupby('waste_group').agg(
                    From_DT=('created_at', 'min'),
                    To_DT=('created_at', 'max'),
                    Avg_Luminance=('TypeB_Luminance', 'mean')
                )
                
                # 4. Calculate strict duration
                event_summary['Duration'] = event_summary['To_DT'] - event_summary['From_DT']
                event_summary['Total Duration (hrs)'] = (event_summary['Duration'].dt.total_seconds() / 3600).round(2)
                
                # 5. FILTER: Keep events >= 30 mins (using 0.45 to catch float math quirks)
                valid_events = event_summary[event_summary['Total Duration (hrs)'] >= 0.45].copy()
                
                if len(valid_events) > 0:
                    # Calculate penalized waste (Subtract 0.5 hours / 30 mins)
                    valid_events['Wasted Hours (Penalty)'] = (valid_events['Total Duration (hrs)'] - 0.5).clip(lower=0).round(2)
                    wasted_hours_total = valid_events['Wasted Hours (Penalty)'].sum()
                    
                    # Format for display table
                    valid_events['Start Date'] = valid_events['From_DT'].dt.strftime('%Y-%m-%d')
                    valid_events['Day'] = valid_events['From_DT'].dt.strftime('%A')
                    valid_events['From Time'] = valid_events['From_DT'].dt.strftime('%H:%M')
                    valid_events['End Date'] = valid_events['To_DT'].dt.strftime('%Y-%m-%d')
                    valid_events['To Time'] = valid_events['To_DT'].dt.strftime('%H:%M')
                    valid_events['Avg Luminance'] = valid_events['Avg_Luminance'].round(0)
                    
                    display_cols = ['Start Date', 'Day', 'From Time', 'End Date', 'To Time', 'Total Duration (hrs)', 'Wasted Hours (Penalty)', 'Avg Luminance']
                    display_waste = valid_events[display_cols].reset_index(drop=True)
                    
                    # --- PREPARE DATA FOR VISUALIZATIONS ---
                    valid_group_ids = valid_events.index
                    plot_df = ghost_calc[ghost_calc['waste_group'].isin(valid_group_ids) & ghost_calc['base_waste_cond']].copy()
                    plot_df['created_at'] = pd.to_datetime(plot_df['created_at'])
                    
                    # Strictly penalized data (Drops first 3 rows / 30 mins)
                    plot_df = plot_df.sort_values(['waste_group', 'created_at'])
                    plot_df_penalty = plot_df.groupby('waste_group').tail(-3).copy()
                    plot_df_penalty['Day_of_Week'] = plot_df_penalty['created_at'].dt.day_name()
                    plot_df_penalty['Hour'] = plot_df_penalty['created_at'].dt.hour
                    
                    # --- UI LAYOUT ---
                    st.markdown("---")
                    
                    col_gauge, col_bar = st.columns([1, 1.5])
                    
                    with col_gauge:
                        st.metric("Total Penalized Wasted Hours", f"{wasted_hours_total:.1f} hrs")
                        max_gauge_val = max(50, wasted_hours_total * 1.5) if wasted_hours_total > 0 else 50
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=wasted_hours_total,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [None, max_gauge_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "#F4B400"}, 
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, max_gauge_val * 0.3], 'color': "#fff9e6"},
                                    {'range': [max_gauge_val * 0.3, max_gauge_val * 0.7], 'color': "#ffedb3"}],
                            }
                        ))
                        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with col_bar:
                        day_dist = plot_df_penalty.groupby('Day_of_Week', observed=False).size().reset_index(name='intervals')
                        day_dist['Penalty Hours'] = (day_dist['intervals'] / 6.0).round(1)
                        waste_by_day = day_dist.sort_values(by='Penalty Hours', ascending=True)
                        
                        if not waste_by_day.empty and waste_by_day['Penalty Hours'].sum() > 0:
                            fig_worst_day = px.bar(
                                waste_by_day, 
                                x='Penalty Hours', 
                                y='Day_of_Week', 
                                orientation='h', 
                                title="Worst Offenders (Penalized Hours by Day)",
                                text_auto='.1f',
                                color='Penalty Hours',
                                color_continuous_scale='YlOrBr' 
                            )
                            fig_worst_day.update_layout(height=320, showlegend=False, coloraxis_showscale=False, margin=dict(l=10, r=20, t=40, b=20))
                            st.plotly_chart(fig_worst_day, use_container_width=True)

                    st.markdown("---")
                    st.markdown("**📅 Lighting Schedule Flaw Heatmap**")
                    st.markdown("Pinpoints exactly when lights are being left on overnight or during weekends.")
                    
                    heatmap_data = plot_df_penalty.groupby(['Day_of_Week', 'Hour'], observed=False).size().reset_index(name='intervals')
                    heatmap_data['Waste_Hours'] = heatmap_data['intervals'] / 6.0
                    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    heatmap_pivot = heatmap_data.pivot(index='Day_of_Week', columns='Hour', values='Waste_Hours').reindex(days_order).fillna(0)
                    
                    for h in range(24):
                        if h not in heatmap_pivot.columns:
                            heatmap_pivot[h] = 0
                    heatmap_pivot = heatmap_pivot[range(24)]
                    
                    fig_heat = px.imshow(
                        heatmap_pivot,
                        labels=dict(x="Hour of Day", y="Day of the Week", color="Hours of Waste"),
                        x=[f"{h:02d}:00" for h in range(24)],
                        y=days_order,
                        color_continuous_scale="YlOrBr",
                        aspect="auto",
                        text_auto=".1f"
                    )
                    fig_heat.update_xaxes(side="top")
                    fig_heat.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=350)
                    st.plotly_chart(fig_heat, use_container_width=True)
                    
                    st.markdown("---")
                    
                    fig_timeline = px.scatter(
                        plot_df, x='created_at', y='TypeB_Luminance', 
                        color_discrete_sequence=['#F4B400'], 
                        title="Luminance Profile During Ghost Lighting Events"
                    )
                    fig_timeline.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    st.markdown("**Detailed Waste Event Log**")
                    st.dataframe(display_waste, use_container_width=True, hide_index=True)
                    
                else:
                    st.success("No sustained ghost lighting detected! Good job turning off the lights.")
            else:
                st.success("No sustained ghost lighting detected! Good job turning off the lights.")

    # --- TAB 2: VENTILATION FAILURE (UNCHANGED) ---
    with tab2:
        st.subheader("Stuffy Air / Ventilation Alert")
        with st.expander("⚙️ View Algorithm Logic & Thresholds"):
            st.markdown("""
            **Condition for Flagging:**
            * **`CO2 30-Min Average >= 1000 ppm`:** Based on ASHRAE 62.1 standards. A 30-minute rolling average filters out brief sensor spikes or transient crowds. Sustained levels above 1000 ppm indicate inadequate fresh air intake.
            """)
            
        if 'TypeB_CO2_ppm' in calc_df.columns:
            # Calculate a 30-minute (3 rows) rolling average to smooth out spikes
            calc_df['CO2_Rolling_Avg'] = calc_df['TypeB_CO2_ppm'].rolling(window=3, min_periods=1).mean()
            
            calc_df['poor_ventilation'] = calc_df['CO2_Rolling_Avg'] >= 1000
            vent_df = calc_df[calc_df['poor_ventilation']]
            vent_hours = len(vent_df) / 6
            
            st.metric("Total Sustained Hours > 1000 ppm", f"{vent_hours:.1f} hrs")
            if vent_hours > 0:
                fig_vent = px.scatter(vent_df, x='created_at', y='CO2_Rolling_Avg', color_discrete_sequence=['darkorange'], title="Sustained High CO2 Events (30-Min Avg)")
                fig_vent.add_hline(y=1000, line_dash="dash", line_color="red", annotation_text="ASHRAE Limit")
                st.plotly_chart(fig_vent, use_container_width=True)
                
                # --- Event Grouping and Logging ---
                calc_df['vent_group'] = (calc_df['poor_ventilation'] != calc_df['poor_ventilation'].shift()).cumsum()
                event_summary = calc_df[calc_df['poor_ventilation']].groupby('vent_group').agg(
                    From_DT=('created_at', 'min'),
                    To_DT=('created_at', 'max'),
                    Avg_CO2=('TypeB_CO2_ppm', 'mean') # Actual average during the incident
                )
                
                # Add 10 mins to account for the width of the final reading
                event_summary['Duration'] = event_summary['To_DT'] - event_summary['From_DT'] + pd.Timedelta(minutes=10)
                
                event_summary['Start Date'] = event_summary['From_DT'].dt.strftime('%Y-%m-%d')
                event_summary['Day'] = event_summary['From_DT'].dt.strftime('%A')
                event_summary['From Time'] = event_summary['From_DT'].dt.strftime('%H:%M')
                event_summary['End Date'] = event_summary['To_DT'].dt.strftime('%Y-%m-%d')
                event_summary['To Time'] = event_summary['To_DT'].dt.strftime('%H:%M')
                event_summary['Total Hours'] = (event_summary['Duration'].dt.total_seconds() / 3600).round(2)
                event_summary['Avg CO2 (ppm)'] = event_summary['Avg_CO2'].round(0)
                
                display_vent = event_summary[['Start Date', 'Day', 'From Time', 'End Date', 'To Time', 'Total Hours', 'Avg CO2 (ppm)']].reset_index(drop=True)
                
                with st.expander("🔍 View Detailed Fault Log"):
                    st.dataframe(display_vent, use_container_width=True)
            else:
                st.success("Air quality is excellent! No sustained stuffy periods detected.")
        else:
            st.warning("Missing CO2 sensor required for this alert.")

    # --- TAB 3: FREEZER Zone (UNCHANGED) ---
    with tab3:
        st.subheader("Overcooling / Freezer Zone")
        with st.expander("⚙️ View Algorithm Logic & Thresholds"):
            st.markdown("""
            **Condition for Flagging:**
            * **`CO2 30-Min Average > 600 ppm`:** The zone is actively and consistently occupied.
            * **`Luminance 30-Min Average > 50`:** Confirms the area is steadily in use.
            * **`Temp 30-Min Average < 21.0°C` (69.8°F):** Drops below the standard ASHRAE 55 thermal comfort baseline for 30+ minutes, indicating the system is actively overcooling the occupants.
            """)
            
        if all(s in calc_df.columns for s in ['TypeB_Luminance', 'TypeB_CO2_ppm', 'TypeB_Temp']):
            # Apply 30-minute rolling averages
            calc_df['CO2_Rolling_Avg'] = calc_df['TypeB_CO2_ppm'].rolling(window=3, min_periods=1).mean()
            calc_df['Lum_Rolling_Avg'] = calc_df['TypeB_Luminance'].rolling(window=3, min_periods=1).mean()
            calc_df['Temp_Rolling_Avg'] = calc_df['TypeB_Temp'].rolling(window=3, min_periods=1).mean()
            
            calc_df['freezer_zone'] = (calc_df['CO2_Rolling_Avg'] > 600) & (calc_df['Lum_Rolling_Avg'] > 50) & (calc_df['Temp_Rolling_Avg'] < 21.0)
            freeze_df = calc_df[calc_df['freezer_zone']]
            freeze_hours = len(freeze_df) / 6
            
            st.metric("Total Sustained Overcooled Hours", f"{freeze_hours:.1f} hrs")
            if freeze_hours > 0:
                fig_freeze = px.scatter(freeze_df, x='created_at', y='Temp_Rolling_Avg', color_discrete_sequence=['blue'], title="Sustained Overcooling Events (30-Min Avg)")
                st.plotly_chart(fig_freeze, use_container_width=True)
                
                # --- Event Grouping and Logging ---
                calc_df['freeze_group'] = (calc_df['freezer_zone'] != calc_df['freezer_zone'].shift()).cumsum()
                event_summary = calc_df[calc_df['freezer_zone']].groupby('freeze_group').agg(
                    From_DT=('created_at', 'min'),
                    To_DT=('created_at', 'max'),
                    Avg_Temp=('TypeB_Temp', 'mean') # Actual average temp during the incident
                )
                
                event_summary['Duration'] = event_summary['To_DT'] - event_summary['From_DT'] + pd.Timedelta(minutes=10)
                
                event_summary['Start Date'] = event_summary['From_DT'].dt.strftime('%Y-%m-%d')
                event_summary['Day'] = event_summary['From_DT'].dt.strftime('%A')
                event_summary['From Time'] = event_summary['From_DT'].dt.strftime('%H:%M')
                event_summary['End Date'] = event_summary['To_DT'].dt.strftime('%Y-%m-%d')
                event_summary['To Time'] = event_summary['To_DT'].dt.strftime('%H:%M')
                event_summary['Total Hours'] = (event_summary['Duration'].dt.total_seconds() / 3600).round(2)
                event_summary['Avg Temp (°C)'] = event_summary['Avg_Temp'].round(2)
                
                display_freeze = event_summary[['Start Date', 'Day', 'From Time', 'End Date', 'To Time', 'Total Hours', 'Avg Temp (°C)']].reset_index(drop=True)
                
                with st.expander("🔍 View Detailed Fault Log"):
                    st.dataframe(display_freeze, use_container_width=True)
            else:
                st.success("No sustained overcooling detected! The thermostat is well balanced.")
        else:
            st.warning("Missing Temp, Luminance, or CO2 sensors required for this alert.")

    # --- TAB 4: SENSOR HEALTH (UNCHANGED) ---
    with tab4:
        st.subheader("Hardware Flatline Detector")
        with st.expander("⚙️ View Algorithm Logic & Thresholds"):
            st.markdown("""
            **Condition for Flagging:**
            * **`Time Duration >= 4 Hours`:** Real-world environmental data always fluctuates slightly. If a sensor reports the *exact* same decimal value for 4 continuous hours or more, it has likely frozen or lost connection to the network.
            """)
            
        flatline_detected = False
        for sensor in sensors:
            # 1. Create a grouping ID that increments every time the value changes
            group_col = f"{sensor}_group"
            calc_df[group_col] = (calc_df[sensor] != calc_df[sensor].shift()).cumsum()
            
            # 2. Extract the first and last timestamp for every identical group
            group_summary = calc_df.groupby(group_col).agg(
                From_DT=('created_at', 'min'),
                To_DT=('created_at', 'max'),
                Frozen_Value=(sensor, 'first')
            )
            
            # 3. Mathematically calculate the exact time difference (duration)
            group_summary['Duration'] = group_summary['To_DT'] - group_summary['From_DT']
            
            # 4. Filter strictly for durations that are 4 hours or longer
            flatlines = group_summary[group_summary['Duration'] >= pd.Timedelta(hours=4)].copy()
            
            if not flatlines.empty:
                flatline_detected = True
                st.error(f"⚠️ **Hardware Fault Detected:** '{sensor}' experienced flatlining.")
                
                # 5. Format the final output table with End Dates and Duration
                flatlines['Start Date'] = flatlines['From_DT'].dt.strftime('%Y-%m-%d')
                flatlines['Day'] = flatlines['From_DT'].dt.strftime('%A')
                flatlines['From Time'] = flatlines['From_DT'].dt.strftime('%H:%M')
                flatlines['End Date'] = flatlines['To_DT'].dt.strftime('%Y-%m-%d')
                flatlines['To Time'] = flatlines['To_DT'].dt.strftime('%H:%M')
                flatlines['Total Hours'] = (flatlines['Duration'].dt.total_seconds() / 3600).round(1)
                
                # Organize columns neatly
                display_df = flatlines[['Start Date', 'Day', 'From Time', 'End Date', 'To Time', 'Total Hours', 'Frozen_Value']].reset_index(drop=True)
                display_df.rename(columns={'Frozen_Value': 'Frozen Value'}, inplace=True)
                
                with st.expander(f"🔍 View summary log for {sensor}"):
                    st.dataframe(display_df, use_container_width=True)
                
        if not flatline_detected:
            st.success("✅ All sensors are actively fluctuating and reporting healthy data streams!")
