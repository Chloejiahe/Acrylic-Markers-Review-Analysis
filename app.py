import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="丙烯调研-双对比版", layout="wide")
st.title("🎨 丙烯颜料市场调研：销量王 vs 潜力股")

# --- 1. 数据加载逻辑 ---
@st.cache_data
def load_excel_data():
    file_map = {
        "kids_sales.xlsx": ("儿童", "销量最高"),
        "kids_trending.xlsx": ("儿童", "趋势最高"),
        "large_capacity_sales.xlsx": ("大容量", "销量最高"),
        "large_capacity_trending.xlsx": ("大容量", "趋势最高")
    }
    combined = []
    for filename, info in file_map.items():
        if os.path.exists(filename):
            try:
                df = pd.read_excel(filename, engine='openpyxl')
                df['category'] = info[0]
                df['data_type'] = info[1]
                # 统一列名：尝试匹配 Content 或 English Content
                target_col = 'Content' if 'Content' in df.columns else ('English Content' if 'English Content' in df.columns else None)
                if target_col:
                    df = df.rename(columns={target_col: 'body'})
                combined.append(df)
            except Exception as e:
                st.error(f"加载 {filename} 失败: {e}")
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

df = load_excel_data()

# --- 2. 侧边栏：核心产品线筛选 ---
st.sidebar.header("📊 核心筛选")
main_cat = st.sidebar.radio("选择调研产品线", ["儿童丙烯", "大容量成人丙烯"])
target_tag = "儿童" if "儿童" in main_cat else "大容量"

# 过滤出当前产品线的数据
cat_df = df[df['category'] == target_tag].copy()
cat_df['body'] = cat_df['body'].fillna('').astype(str)

# --- 3. 页面布局：双支线对比分析 ---
st.header(f"🔍 {main_cat}：市场基本盘 vs 新兴趋势")

# 定义分析关键词
high_kws = {"色彩/覆盖力": "vibrant|bright|coverage|pigment", "包装/收纳": "case|box|storage|organized", "礼品属性": "gift|present|grand"}
pain_kws = {"白色缺失": "white|ran out|more white", "容易干涸": "dry|dried|stuck|clog", "物流/破损": "leak|mess|broken"}

def get_analysis(data):
    results = {}
    for label, kw in {**high_kws, **pain_kws}.items():
        results[label] = data['body'].str.contains(kw, case=False, na=False).sum()
    return pd.Series(results)

# 创建两个并排的列
col_sales, col_trend = st.columns(2)

with col_sales:
    st.subheader("🏆 销量最高 (Top 10)")
    sales_data = cat_df[cat_df['data_type'] == "销量最高"]
    st.write(f"样本量: {len(sales_data)} 条评论")
    
    # 满意点与痛点图表
    st.bar_chart(get_analysis(sales_data))
    
    with st.expander("查看销量王典型评论"):
        st.write(sales_data['body'].head(10))

with col_trend:
    st.subheader("🚀 趋势最高 (Trending)")
    trend_data = cat_df[cat_df['data_type'] == "趋势最高"]
    st.write(f"样本量: {len(trend_data)} 条评论")
    
    # 满意点与痛点图表
    st.bar_chart(get_analysis(trend_data))
    
    with st.expander("查看趋势黑马典型评论"):
        st.write(trend_data['body'].head(10))

# --- 4. 深度洞察对比 ---
st.divider()
st.subheader("💡 跨维度洞察：我们学到了什么？")

obs_col1, obs_col2 = st.columns(2)
with obs_col1:
    st.info("**销量款告诉我们‘底线’**：\n\n这些成熟产品最常被吐槽的问题，就是我们必须解决的‘入场券’（例如：大容量款必须多配白色）。")
with obs_col2:
    st.warning("**趋势款告诉我们‘机会’**：\n\n新爆款往往是因为解决了一个特定痛点（如：儿童款带了收纳包）而迅速蹿红，这是我们要抄的‘近道’。")
