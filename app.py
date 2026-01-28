import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- 页面配置 ---
st.set_page_config(page_title="丙烯调研看板", layout="wide")

# --- 数据加载函数 ---
@st.cache_data
def load_raw_data():
    # 建立文件与板块的对应关系
    # 这里的键名必须和你 GitHub 上的文件名完全一致
    data_map = {
        "kids_sales.xlsx": ("儿童丙烯", "🔥 高销量 (Top 10)"),
        "kids_trending.xlsx": ("儿童丙烯", "📈 高增长趋势"),
        "large_capacity_sales.xlsx": ("大容量丙烯", "🔥 高销量 (Top 10)"),
        "large_capacity_trending.xlsx": ("大容量丙烯", "📈 高增长趋势")
    }
    
    combined = []
    for filename, info in data_map.items():
        if os.path.exists(filename):
            try:
                df = pd.read_excel(filename, engine='openpyxl')
                df['main_category'] = info[0]  # 第一层级
                df['sub_type'] = info[1]       # 第二层级
                combined.append(df)
            except Exception as e:
                st.sidebar.error(f"读取 {filename} 失败: {e}")
                
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

# 加载数据
df = load_raw_data()

# --- 网站结构设计 ---

# 1. 顶部标题
st.title("🎨 丙烯颜料市场竞争调研看板")
st.caption("数据源：Amazon 评论数据 (销量 Top 10 与 增长趋势 Top 10)")

# 2. 侧边栏 - 第一层级导航：产品大类
st.sidebar.header("📂 核心板块选择")
selected_main = st.sidebar.radio(
    "请选择调研产品线：",
    ["儿童丙烯", "大容量丙烯"]
)

# 过滤出该大类下的数据
filtered_df = df[df['main_category'] == selected_main]

# 3. 主界面 - 第二层级布局：销量 vs 趋势
if not filtered_df.empty:
    st.header(f"📍 当前板块：{selected_main}")
    
    # 使用两列布局，分别放置“高销量”和“高增长趋势”
    col_sales, col_trend = st.columns(2)
    
    with col_sales:
        st.subheader("🔥 销量最高 (Best Sellers)")
        sales_data = filtered_df[filtered_df['sub_type'].str.contains("销量")]
        if not sales_data.empty:
            st.info(f"已加载 {len(sales_data)} 条原始评论")
            # 仅展示前50条数据预览，不进行任何分析
            st.dataframe(sales_data, use_container_width=True)
        else:
            st.warning("暂无销量数据文件")

    with col_trend:
        st.subheader("📈 增长趋势 (Trending Stars)")
        trend_data = filtered_df[filtered_df['sub_type'].str.contains("趋势")]
        if not trend_data.empty:
            st.info(f"已加载 {len(trend_data)} 条原始评论")
            # 仅展示前50条数据预览，不进行任何分析
            st.dataframe(trend_data, use_container_width=True)
        else:
            st.warning("暂无趋势数据文件")
            
else:
    st.error("未检测到任何数据文件，请检查 GitHub 根目录下的 .xlsx 文件。")
    st.write("当前目录下的文件有：", os.listdir('.'))
