import streamlit as st
import pandas as pd
import plotly.express as px

# 1. 页面配置
st.set_page_config(page_title="丙烯调研报告", layout="wide")
st.title("🎨 丙烯颜料评论分析看板")

# 2. 核心加载逻辑
@st.cache_data
def load_all_data():
    # 严格按照你截图中的文件名（注意空格和后缀）
    file_info = {
        "kids_sales.xlsx - Sheet1.csv": ("儿童", "销量Top10"),
        "kids_trending.xlsx - Sheet1.csv": ("儿童", "趋势Top10"),
        "large_capacity_sales.xlsx - Sheet2.csv": ("大容量", "销量Top10"),
        "large_capacity_trending.xlsx - Sheet1.csv": ("大容量", "趋势Top10")
    }
    
    all_dfs = []
    
    for filename, info in file_info.items():
        try:
            # 加上 encoding='utf-8' 防止乱码报错
            temp_df = pd.read_csv(filename, encoding='utf-8')
            temp_df['category'] = info[0]
            temp_df['data_type'] = info[1]
            # 统一列名清洗：确保 body 列存在且没空格
            temp_df.columns = temp_df.columns.str.strip()
            all_dfs.append(temp_df)
        except Exception as e:
            st.sidebar.error(f"无法读取 {filename}: {e}")
            
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

df = load_all_data()

# --- 核心诊断：如果 df 为空，直接提示原因并停止运行 ---
if df.empty:
    st.error("🚨 报错啦！当前仓库内一个数据文件都没读到。")
    st.info("请检查：GitHub 上的文件名是否真的包含 '.xlsx - Sheet1.csv' 这种后缀？如果文件名改了，代码里的字典也要改。")
    st.stop()

# 3. 侧边栏选择
st.sidebar.header("数据筛选")
cat_choice = st.sidebar.radio("选择产品线", ["儿童丙烯", "大容量款"])

# 4. 数据过滤逻辑
# 这里用 contains 防止名称不完全匹配
mask = df['category'].str.contains("儿童") if "儿童" in cat_choice else df['category'].str.contains("大容量")
selected_df = df[mask]

# 5. 分析模块（满意点/痛点）
st.subheader(f"🔍 {cat_choice} 分析结果")

# 定义关键词
pain_kws = {"白色不足": "white|empty|not enough|more white", "干燥堵塞": "dry|clog|stuck", "包装漏液": "leak|mess|spilled"}
hi_kws = {"收纳好评": "box|case|storage|organizer", "色彩好": "vibrant|bright|pigment", "送礼": "gift|daughter|son"}

def get_counts(data, kws):
    res = {}
    for k, v in kws.items():
        res[k] = data['body'].str.contains(v, case=False, na=False).sum()
    return pd.Series(res)

col1, col2 = st.columns(2)
with col1:
    st.success("✅ 满意点统计")
    st.bar_chart(get_counts(selected_df, hi_kws))
with col2:
    st.error("❌ 痛点统计")
    st.bar_chart(get_counts(selected_df, pain_kws))

st.write("---")
st.write("📂 **数据预览 (前 5 条):**")
st.dataframe(selected_df[['category', 'data_type', 'body']].head())
