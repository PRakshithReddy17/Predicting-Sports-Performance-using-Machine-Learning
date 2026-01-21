import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="REAL TIME MONITORING SPORTS ANALYSIS", layout="wide")

@st.cache_data
def load_data():
    data = {
        'Player': [
            'Sai Sudharsan', 'Shubman Gill', 'Suryakumar Yadav', 'Virat Kohli',
            'Ruturaj Gaikwad', 'KL Rahul', 'Sanju Samson', 'Rinku Singh',
            'Heinrich Klaasen', 'Nicholas Pooran', 'Travis Head', 'Abhishek Sharma'
        ],
        'Matches': [15, 15, 16, 15, 15, 14, 14, 13, 12, 13, 12, 12],
        'Runs': [759, 650, 717, 657, 590, 560, 552, 480, 462, 440, 567, 521],
        'Average': [54.21, 50.00, 65.18, 54.75, 49.16, 46.67, 47.83, 40.00, 42.00, 39.50, 51.54, 43.41],
        'StrikeRate': [156.17, 155.87, 167.91, 144.71, 145.09, 139.28, 155.31, 159.87, 180.52, 172.18, 191.23, 189.43],
        'Fours': [85, 70, 80, 60, 75, 68, 65, 55, 48, 45, 75, 72],
        'Sixes': [25, 22, 28, 20, 21, 18, 19, 15, 25, 23, 32, 31]
    }
    return pd.DataFrame(data)


df = load_data()

st.sidebar.title("Controls")
plot_choice = st.sidebar.selectbox("Choose plot type:",
                                   [
                                       'Overview Table', 'Scatter', 'Bar (Top N)', 'Line (Runs by Player)',
                                       'Histogram', 'Boxplot', 'Violin', 'Density (KDE)', 'Pairplot',
                                       'Correlation Heatmap', 'Radar Chart', 'Actual vs Predicted (CV)'
                                   ])

x_col = st.sidebar.selectbox('X axis', options=['Matches', 'Average', 'StrikeRate', 'Fours', 'Sixes'], index=0)

y_col = st.sidebar.selectbox('Y axis', options=['Runs', 'Average', 'StrikeRate', 'Fours', 'Sixes'], index=0)

color_col = st.sidebar.selectbox('Color (optional)', options=[None, 'Player'], index=0)

bins = st.sidebar.slider('Bins (for histograms)', min_value=5, max_value=30, value=12)

top_n = st.sidebar.slider('Top N players (bar plot)', min_value=3, max_value=len(df), value=8)

show_table = st.sidebar.checkbox('Show raw table', value=True)

st.title("🏏 REAL TIME MONITORING SPORTS ANALYSIS")
st.markdown("Use the sidebar to switch plots and control options. This demo shows many common chart types for exploratory data analysis.")

if show_table:
    st.subheader("Dataset")
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), file_name='ipl2025_players.csv')

def st_plt(fig):
    st.pyplot(fig)

if plot_choice == 'Overview Table':
    st.subheader('Summary statistics')
    st.write(df.describe())

elif plot_choice == 'Scatter':
    st.subheader(f"Scatter: {y_col} vs {x_col}")
    fig = px.scatter(df, x=x_col, y=y_col, hover_name='Player', size='Matches', color=color_col if color_col else None,
                     title=f"{y_col} vs {x_col}")
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == 'Bar (Top N)':
    st.subheader(f"Bar: Top {top_n} players by Runs")
    top_df = df.sort_values('Runs', ascending=False).head(top_n)
    fig = px.bar(top_df, x='Player', y='Runs', hover_data=['Matches','Average','StrikeRate','Fours','Sixes'],
                 title=f"Top {top_n} Run-scorers")
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == 'Line (Runs by Player)':
    st.subheader('Line chart — Runs by player (sorted)')
    plot_df = df.sort_values('Runs', ascending=False).reset_index(drop=True)
    fig = px.line(plot_df, x='Player', y='Runs', markers=True, title='Runs by Player (descending)')
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == 'Histogram':
    st.subheader(f"Histogram of {y_col}")
    fig = px.histogram(df, x=y_col, nbins=bins, title=f'Histogram of {y_col}')
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == 'Boxplot':
    st.subheader(f"Boxplot of {y_col}")
    fig = px.box(df, y=y_col, points='all', hover_name='Player', title=f'Boxplot: {y_col}')
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == 'Violin':
    st.subheader(f"Violin plot of {y_col}")
    fig = px.violin(df, y=y_col, box=True, points='all', hover_name='Player', title=f'Violin: {y_col}')
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == 'Density (KDE)':
    st.subheader(f"Density plot (KDE) of {y_col}")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.kdeplot(df[y_col], fill=True, ax=ax)
    ax.set_title(f'KDE: {y_col}')
    st_plt(fig)

elif plot_choice == 'Pairplot':
    st.subheader('Pairplot of numeric features')
    numeric_cols = ['Matches','Runs','Average','StrikeRate','Fours','Sixes']
    pairgrid = sns.pairplot(df[numeric_cols], diag_kind='kde')
    st.pyplot(pairgrid.fig)

elif plot_choice == 'Correlation Heatmap':
    st.subheader('Correlation heatmap')
    corr = df[['Matches','Runs','Average','StrikeRate','Fours','Sixes']].corr()
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlation')
    st_plt(fig)

elif plot_choice == 'Radar Chart':
    st.subheader('Radar chart (normalized)')
    player = st.selectbox('Select player for radar', df['Player'].tolist())
    feat_cols = ['Matches','Runs','Average','StrikeRate','Fours','Sixes']
    norm = (df[feat_cols] - df[feat_cols].min()) / (df[feat_cols].max() - df[feat_cols].min())
    r = norm[df['Player']==player].iloc[0].values.tolist()
    theta = feat_cols
    r = r + [r[0]]
    theta = theta + [theta[0]]
    fig = px.line_polar(r=r, theta=theta, line_close=True, title=f'Radar: {player}')
    fig.update_traces(fill='toself')
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == 'Actual vs Predicted (CV)':
    st.subheader('Train a model and show cross-validated Actual vs Predicted')
    features = st.multiselect('Select features for model', options=['Matches','Average','StrikeRate','Fours','Sixes'], default=['Matches','Average','StrikeRate','Fours','Sixes'])
    if len(features) < 1:
        st.error('Select at least one feature')
    else:
        X = df[features]
        y = df['Runs']
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_cv = cross_val_predict(model, X, y, cv=kf)
        mae = mean_absolute_error(y, y_pred_cv)
        rmse = mean_squared_error(y, y_pred_cv, squared=False)
        r2 = r2_score(y, y_pred_cv)
        st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
        fig = px.scatter(x=y, y=y_pred_cv, labels={'x':'Actual Runs','y':'Predicted Runs'}, hover_name=df['Player'], title='Actual vs Predicted (CV)')
        fig.add_shape(type='line', x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(dash='dash'))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("**Tips:** Use different plot types and hover to explore player-level insights. For small datasets prefer cross-validation and avoid overfitting.")

@st.cache_data
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="openpyxl")
    df.to_excel(writer, index=False, sheet_name='data')
    writer.close()
    return output.getvalue()

excel_data = to_excel(df)
st.download_button(label='Download Excel', data=excel_data, file_name='ipl2025_players.xlsx')

