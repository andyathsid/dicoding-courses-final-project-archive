import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import plotly.subplots as sp

# Helper functions
def create_hourly_patterns_df(df):
    hourly_patterns = df.groupby('hr').agg({
        'casual': 'mean',
        'registered': 'mean',
        'cnt': 'mean'
    }).reset_index()
    return hourly_patterns

def create_daily_patterns_df(df):
    daily_patterns = df.groupby('weekday').agg({
        'casual': 'mean',
        'registered': 'mean',
        'cnt': 'mean'
    }).reset_index()
    daily_patterns['day_name'] = daily_patterns['weekday'].map({
        0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
        4: 'Thursday', 5: 'Friday', 6: 'Saturday'
    })
    return daily_patterns

def create_weather_impact_df(df):
    weather_impact = df.groupby('weathersit').agg({
        'casual': 'mean',
        'registered': 'mean',
        'cnt': 'mean'
    }).reset_index()
    weather_impact['weather_desc'] = weather_impact['weathersit'].map({
        1: 'Clear',
        2: 'Mist + Cloudy',
        3: 'Light Snow/Rain',
        4: 'Heavy Rain/Snow'
    })
    return weather_impact

def create_monthly_trends_df(df):
    df['ym'] = pd.to_datetime(df['dteday']).dt.to_period('M')
    monthly_trends = df.groupby(['ym', 'yr']).agg({
        'cnt': 'mean'
    }).reset_index()
    monthly_trends['ym'] = monthly_trends['ym'].astype(str)
    return monthly_trends

def create_lag_analysis_df(df):
    """Create dataframe for lag analysis of weather impact"""
    
    df['cnt_lag1'] = df['cnt'].shift(-1)
    df['cnt_lag2'] = df['cnt'].shift(-2)
    df['casual_lag1'] = df['casual'].shift(-1)
    df['casual_lag2'] = df['casual'].shift(-2)
    df['registered_lag1'] = df['registered'].shift(-1)
    df['registered_lag2'] = df['registered'].shift(-2)

    
    # Filter 
    bad_weather_df = df[df['weathersit'] >= 3].copy()
    return bad_weather_df

@st.cache_data
def load_data():
    hour_df = pd.read_csv('../data/bike-sharing-dataset/hour.csv')
    day_df = pd.read_csv('../data/bike-sharing-dataset/day.csv')
    
    hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    
    return hour_df, day_df

st.set_page_config(
    page_title="Bike Sharing Analysis Dashboard",
    page_icon="🚲",
    layout="wide"
)

with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Project Information
        - This is part of Dicoding Academy's Data Analysis Course final project
        - **Author:** Andakara Athaya Sidiq
        - **Dicoding ID:** andyathsid
        """)
    with col2:
        st.markdown("""
        ### Quick Links
        - [GitHub](https://github.com/andyathsid/dicoding-courses-final-project-archive/tree/c94653e35c2671c4c9b23ae0c43b6e292d97b7a0/learn_data_analysis_with_python)
        - [Data Source](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
        """)
    
hour_df, day_df = load_data()

st.title("🚲 Bike Sharing Analysis Dashboard")
st.write("This dashboard provides comprehensive insights into bike sharing patterns and trends, analyzing temporal patterns, weather impacts, and user behavior.")

# Filters
with st.sidebar:
    st.header("Filters")
    
    # filter tanggal
    date_range = st.date_input(
        "Select Date Range",
        value=(hour_df['dteday'].min(), hour_df['dteday'].max()),
        min_value=hour_df['dteday'].min().date(),
        max_value=hour_df['dteday'].max().date()
    )
    
    # Filter cuaca
    weather_options = {
        1: "Clear",
        2: "Mist + Cloudy",
        3: "Light Snow/Rain",
        4: "Heavy Rain/Snow"
    }
    selected_weather = st.multiselect(
        "Select Weather Conditions",
        options=list(weather_options.keys()),
        default=list(weather_options.keys()),
        format_func=lambda x: weather_options[x]
    )

filtered_hour_df = hour_df[
    (hour_df['dteday'].dt.date >= date_range[0]) &
    (hour_df['dteday'].dt.date <= date_range[1]) &
    (hour_df['weathersit'].isin(selected_weather))
]

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    total_rides = filtered_hour_df['cnt'].sum()
    st.metric("Total Rides", f"{total_rides:,}")
with col2:
    avg_daily_rides = filtered_hour_df.groupby('dteday')['cnt'].sum().mean()
    st.metric("Average Daily Rides", f"{avg_daily_rides:,.0f}")
with col3:
    peak_hour_rides = filtered_hour_df.groupby('hr')['cnt'].mean().max()
    st.metric("Peak Hour Average", f"{peak_hour_rides:,.0f}")

st.header("📊 Peak Period Analysis")
st.write("""
Analyze rental patterns throughout different time periods, showing:
- Daily patterns: Peak hours showing commute times for registered users vs leisure hours for casual users
- Weekly patterns: Weekday vs weekend preferences between user types
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Hourly Patterns")
    hourly_patterns = create_hourly_patterns_df(filtered_hour_df)

    fig_hourly = go.Figure()

    fig_hourly.add_trace(go.Scatter(
        x=hourly_patterns['hr'],
        y=hourly_patterns['casual'],
        name='Casual Users',
        mode='lines+markers',
        line=dict(width=3, color='#ff7f0e'),
        marker=dict(size=8)
    ))

    fig_hourly.add_trace(go.Scatter(
        x=hourly_patterns['hr'],
        y=hourly_patterns['registered'],
        name='Registered Users',
        mode='lines+markers',
        line=dict(width=3, color='#1f77b4'),
        marker=dict(size=8)
    ))

    # Modify annotations to be more compact for smaller plot
    hourly_annotations = [
        dict(x=8, y=hourly_patterns.loc[8, 'registered'],
             text="8 AM Peak",
             showarrow=True,
             arrowhead=1,
             font_size=10),
        dict(x=17, y=hourly_patterns.loc[17, 'registered'],
             text="5 PM Peak",
             showarrow=True,
             arrowhead=1,
             font_size=10),
    ]

    fig_hourly.update_layout(
        title='Average Rentals by Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Average Rentals',
        hovermode='x unified',
        annotations=hourly_annotations,
        xaxis=dict(
            tickmode='array',
            ticktext=['12 AM', '4 AM', '8 AM', '12 PM', '4 PM', '8 PM'],
            tickvals=[0, 4, 8, 12, 16, 20],
            tickangle=45
        ),
        height=400,
        margin=dict(t=30, l=30, r=30, b=30),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

with col2:
    st.subheader("Daily Patterns")
    daily_patterns = create_daily_patterns_df(filtered_hour_df)

    fig_daily = go.Figure()

    fig_daily.add_trace(go.Scatter(
        x=daily_patterns['day_name'],
        y=daily_patterns['casual'],
        name='Casual Users',
        mode='lines+markers',
        line=dict(width=3, color='#ff7f0e'),
        marker=dict(size=8)
    ))

    fig_daily.add_trace(go.Scatter(
        x=daily_patterns['day_name'],
        y=daily_patterns['registered'],
        name='Registered Users',
        mode='lines+markers',
        line=dict(width=3, color='#1f77b4'),
        marker=dict(size=8)
    ))

    # Simplified annotations for smaller plot
    daily_annotations = [
        dict(x='Sunday', y=daily_patterns[daily_patterns['day_name']=='Sunday']['casual'].iloc[0],
             text="Weekend Peak",
             showarrow=True,
             arrowhead=1,
             font_size=10),
        dict(x='Friday', y=daily_patterns[daily_patterns['day_name']=='Friday']['registered'].iloc[0],
             text="Weekday Peak",
             showarrow=True,
             arrowhead=1,
             font_size=10)
    ]

    fig_daily.update_layout(
        title='Average Rentals by Day',
        xaxis_title='Day of Week',
        yaxis_title='Average Rentals',
        hovermode='x unified',
        annotations=daily_annotations,
        height=400,
        margin=dict(t=30, l=30, r=30, b=30),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    st.plotly_chart(fig_daily, use_container_width=True)
    
with st.expander("💡 Key Insights: Peak Period Analysis"):
    st.markdown("""
    1. **Daily Patterns**:
        - Registered Users:
            * Morning Peak (8 AM): ~300 rentals/hour (commute to work)
            * Evening Peak (5 PM): ~350 rentals/hour (return commute)
        - Casual Users: Single peak (2 PM) ~100 rentals/hour
        - Lowest Activity: 4 AM (<20 rentals/hour)

    2. **Weekly Patterns**:
        - Weekdays (Monday-Friday):
            * Dominated by registered users (>3000 rentals/day)
            * Peak on Fridays (~4000 rentals)
        - Weekends:
            * Significant increase in casual users (>1400 rentals/day)
            * More balanced distribution between user types
    """)

# Weather impact
st.header("🌤️ Weather Impact Analysis")
st.write("""
Understand how different weather conditions affect bike rentals, showing the resilience of different user types
to adverse weather conditions.
""")
weather_impact = create_weather_impact_df(filtered_hour_df)

st.header("Weather Impact Analysis")
weather_impact = filtered_hour_df.groupby('weathersit').agg({
    'casual': 'mean',
    'registered': 'mean',
    'cnt': 'mean'
}).reset_index()

weather_labels = {
    1: 'Clear/Partly Cloudy',
    2: 'Mist/Cloudy',
    3: 'Light Rain/Snow',
    4: 'Heavy Rain/Snow'
}
weather_impact['weather_condition'] = weather_impact['weathersit'].map(weather_labels)

fig_weather = px.bar(weather_impact, 
                    x='weather_condition',
                    y=['casual', 'registered'],
                    title='Impact of Weather Conditions on Bike Rentals',
                    labels={
                        'weather_condition': 'Weather Condition',
                        'value': 'Average Daily Rentals',
                        'variable': 'User Type'
                    },
                    template='plotly_white',
                    barmode='group')

baseline_casual = weather_impact.loc[0, 'casual']
baseline_registered = weather_impact.loc[0, 'registered']

fig_weather.update_layout(
    legend_title_text='User Type',
    showlegend=True,
    xaxis_title='Weather Condition',
    yaxis_title='Average Daily Rentals',
    bargap=0.2,
    height=600
)

fig_weather.update_traces(
    marker_color='#1f77b4',
    selector=dict(name='registered')
)
fig_weather.update_traces(
    marker_color='#ff7f0e',
    selector=dict(name='casual')
)

st.plotly_chart(fig_weather, use_container_width=True)

with st.expander("💡 Key Insights: Weather Impact Analysis"):
    st.markdown("""
    1. **Tiered Impact on Rentals**:
        - Clear/Partly Cloudy:
            * Highest rental rates
            * Registered: ~3,600 rentals/day
            * Casual: ~850 rentals/day
        - Misty/Cloudy:
            * Registered: ~15% decrease (~3,000 rentals/day)
            * Casual: ~20% decrease (~680 rentals/day)

    2. **User Sensitivity**:
        - Casual Users:
            * More sensitive to weather changes
            * ~70% decrease during light rain/snow
        - Registered Users:
            * More resilient to adverse weather
            * ~40% decrease during light rain/snow
    """)

# YoY Analysis
st.header("📈 Monthly Trends")
st.write("""
Track the year-over-year growth and seasonal patterns in bike rentals, comparing 2011 vs 2012 performance
and identifying long-term trends.
""")
monthly_trends = create_monthly_trends_df(filtered_hour_df)
monthly_trends = filtered_hour_df.groupby(['yr', 'mnth']).agg({
    'casual': 'mean',
    'registered': 'mean',
    'cnt': 'mean'
}).reset_index()

nama_bulan = {
    1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'Mei', 6:'Jun',
    7:'Jul', 8:'Agu', 9:'Sep', 10:'Okt', 11:'Nov', 12:'Des'
}
monthly_trends['bulan'] = monthly_trends['mnth'].map(nama_bulan)
monthly_trends['tahun'] = monthly_trends['yr'].map({0: '2011', 1: '2012'})
monthly_trends['periode'] = monthly_trends.apply(lambda x: f"{x['bulan']} {x['tahun']}", axis=1)

pertumbuhan_tahunan = {
    'casual': ((monthly_trends[monthly_trends['yr']==1]['casual'].mean() / 
                monthly_trends[monthly_trends['yr']==0]['casual'].mean()) - 1) * 100,
    'registered': ((monthly_trends[monthly_trends['yr']==1]['registered'].mean() / 
                    monthly_trends[monthly_trends['yr']==0]['registered'].mean()) - 1) * 100
}

fig_monthly = go.Figure()

fig_monthly.add_trace(go.Scatter(
    x=monthly_trends['periode'],
    y=monthly_trends['casual'],
    name='Casual Users',
    mode='lines+markers',
    line=dict(width=3, color='#ff7f0e'),
    marker=dict(size=8)
))

fig_monthly.add_trace(go.Scatter(
    x=monthly_trends['periode'],
    y=monthly_trends['registered'],
    name='Registered Users',
    mode='lines+markers',
    line=dict(width=3, color='#1f77b4'),
    marker=dict(size=8)
))

annotations = [
    dict(x='Jun 2012',
         y=monthly_trends['registered'].max() * 0.6,
         text=f"Year-over-Year Growth:<br>Registered: {pertumbuhan_tahunan['registered']:.1f}%<br>Casual: {pertumbuhan_tahunan['casual']:.1f}%",
         showarrow=False,
         bgcolor='rgba(255,255,255,0.8)')
]

fig_monthly.update_layout(
    title='Monthly Rental Trends by User Type (2011-2012)',
    xaxis_title='Month-Year',
    yaxis_title='Average Daily Rentals',
    hovermode='x unified',
    annotations=annotations,
    height=600,
    showlegend=True,
    legend_title_text='User Type',
    xaxis_tickangle=45,
    template='plotly_white'
)

fig_monthly.add_vline(
    x=11.5,  # December 2011
    line_width=2,
    line_dash="dash",
    line_color="gray",
    annotation_text="Year Change",
    annotation_position="top right"
)

st.plotly_chart(fig_monthly, use_container_width=True)

with st.expander("💡 Key Insights: Monthly Trends Analysis"):
    st.markdown("""
    1. **Seasonal Patterns**:
        - Peak: Summer season (Jun-Aug)
        - Lowest: Winter season (Dec-Feb)
        - Registered users show more stable patterns

    2. **Year-over-Year Growth**:
        - Casual users: ~62% growth
        - Registered users: ~39% growth
        - Seasonal patterns remain strong despite growth
        - Weather impacts consistent across both years
    """)

# Lag Analysis
st.header("🔄 Weather Recovery Analysis")
st.write("""
Examine how quickly rental patterns recover after bad weather days, showing the resilience and behavior
patterns of different user types.
""")
lag_df = create_lag_analysis_df(day_df)

fig_lag = sp.make_subplots(
    rows=2, cols=2,
    subplot_titles=('Casual Users: Next Day Effect',
                   'Casual Users: Two Days After',
                   'Registered Users: Next Day Effect',
                   'Registered Users: Two Days After'),
    vertical_spacing=0.2,
    horizontal_spacing=0.15
)

# Casual Users - Next Day
fig_lag.add_trace(
    go.Scatter(
        x=lag_df['casual'],
        y=lag_df['casual_lag1'],
        mode='markers',
        name='Next Day',
        marker=dict(color='#ff7f0e'),
        text=lag_df['dteday']
    ),
    row=1, col=1
)

# Casual Users - Two Days After
fig_lag.add_trace(
    go.Scatter(
        x=lag_df['casual'],
        y=lag_df['casual_lag2'],
        mode='markers',
        name='Two Days After',
        marker=dict(color='#ff7f0e', symbol='square'),
        text=lag_df['dteday']
    ),
    row=1, col=2
)

# Registered Users - Next Day
fig_lag.add_trace(
    go.Scatter(
        x=lag_df['registered'],
        y=lag_df['registered_lag1'],
        mode='markers',
        name='Next Day',
        marker=dict(color='#1f77b4'),
        text=lag_df['dteday']
    ),
    row=2, col=1
)

# Registered Users - Two Days After
fig_lag.add_trace(
    go.Scatter(
        x=lag_df['registered'],
        y=lag_df['registered_lag2'],
        mode='markers',
        name='Two Days After',
        marker=dict(color='#1f77b4', symbol='square'),
        text=lag_df['dteday']
    ),
    row=2, col=2
)

for row in [1, 2]:
    for col in [1, 2]:
        fig_lag.add_trace(
            go.Scatter(
                x=[0, lag_df['cnt'].max()],
                y=[0, lag_df['cnt'].max()],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                showlegend=False
            ),
            row=row, col=col
        )

fig_lag.update_layout(
    height=900,
    width=1200,
    title_text='Recovery of Average Rentals After Bad Weather Days',
    showlegend=True,
    template='plotly_white',
    margin=dict(t=100)
)

fig_lag.update_yaxes(title_text="Penyewaan Hari Berikutnya", row=1, col=1, range=[0, 2700])
fig_lag.update_yaxes(title_text="Penyewaan Dua Hari Setelahnya", row=1, col=2, range=[0, 2000])
fig_lag.update_xaxes(title_text="Penyewaan pada Hari dengan Cuaca Buruk", row=1, col=1, range=[0, 1500])
fig_lag.update_xaxes(title_text="Penyewaan pada Hari dengan Cuaca Buruk", row=1, col=2, range=[0, 1500])

st.plotly_chart(fig_lag, use_container_width=True)

with st.expander("💡 Key Insights: Weather Recovery Analysis"):
    st.markdown("""
    1. **Casual Users Recovery Effect**:
        - Next Day:
            * 40-60% increase from bad weather day
            * Shows postponement of recreational activities
        - Two Days After:
            * Returns to normal levels
            * Higher variation in recovery

    2. **Registered Users Recovery Effect**:
        - Next Day:
            * Moderate increase (20-30%)
            * More consistent rebound pattern
        - Two Days After:
            * Almost completely normal
            * Shows usage stability
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Data source: Capital Bikeshare system, Washington D.C., USA (2011-2012)</p>
    <p>Created by Andakara Athaya Sidiq</p>
</div>
""", unsafe_allow_html=True)