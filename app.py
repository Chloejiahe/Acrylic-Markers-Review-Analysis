import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="丙烯调研报告", layout="wide")
st.title("🎨 丙烯颜料市场调研分析看板")

# --- 1. 数据加载逻辑 (适配 XLSX) ---
@st.cache_data
def load_excel_data():
    # 建立文件名与分类的映射
    file_map = {
        "kids_sales.xlsx": ("儿童", "销量Top10"),
        "kids_trending.xlsx": ("儿童", "趋势Top10"),
        "large_capacity_sales.xlsx": ("大容量", "销量Top10"),
        "large_capacity_trending.xlsx": ("大容量", "趋势Top10")
    }
    
    combined = []
    for filename, info in file_map.items():
        if os.path.exists(filename):
            try:
                # 使用 openpyxl 引擎读取 Excel
                df = pd.read_excel(filename, engine='openpyxl')
                df['category'] = info[0]
                df['data_type'] = info[1]
                
                # 统一列名：将 'Content' 或 'English Content' 统一为 'body'
                if 'Content' in df.columns:
                    df = df.rename(columns={'Content': 'body'})
                elif 'English Content' in df.columns:
                    df = df.rename(columns={'English Content': 'body'})
                
                combined.append(df)
                st.sidebar.success(f"✅ 加载成功: {filename}")
            except Exception as e:
                st.sidebar.error(f"❌ 读取 {filename} 失败: {e}")
    
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

df = load_excel_data()

# --- 2. 异常处理 ---
if df.empty:
    st.error("🚨 还是没读到数据！")
    st.write("当前检测到的文件：", os.listdir('.'))
    st.stop()

# --- 3. 业务看板界面 ---
st.sidebar.divider()
choice = st.sidebar.radio("选择产品线", ["儿童丙烯", "大容量款"])
target = "儿童" if "儿童" in choice else "大容量"
selected_df = df[df['category'] == target].copy()

# 确保评论列是字符串
selected_df['body'] = selected_df['body'].fillna('').astype(str)

tab1, tab2, tab3 = st.tabs(["📊 满意点与痛点", "👤 用户画像", "💡 调研建议"])

with tab1:
    col1, col2 = st.columns(2)
    # 定义匹配词库
    high_kws = {"色彩覆盖力": "vibrant|bright|coverage|opacity|pigment", "收纳盒/包装": "case|box|storage|organized", "顺滑好用": "easy|flow|smooth|marker"}
    pain_kws = {"白色颜料不足": "white|ran out|more white|extra white", "干涸/堵塞": "dry|dried|stuck|clog", "漏液": "leak|mess|spilled"}

    def get_counts(data, kw_dict):
        return pd.Series({k: data['body'].str.contains(v, case=False, na=False).sum() for k, v in kw_dict.items()})

    with col1:
        st.success("✅ 满意点统计")
        st.bar_chart(get_counts(selected_df, high_kws))
    with col2:
        st.error("❌ 痛点统计")
        st.bar_chart(get_counts(selected_df, pain_kws))

with tab2:
    st.subheader("谁在买？（用户画像）")
    persona_kws = {"家长/送礼": "gift|grand|child|son|daughter", "专业/画师": "artist|professional|mural|canvas|rock"}
    p_counts = get_counts(selected_df, persona_kws)
    st.plotly_chart(px.pie(values=p_counts.values, names=p_counts.index, hole=0.4))

with tab3:
    st.subheader("市场行动建议")
    if target == "儿童":
        st.info("儿童款调研结论：**'Case' (收纳盒)** 是核心竞争力。用户反馈这是极佳的生日/节日礼物。建议增加外盒的趣味性设计。")
    else:
        st.warning("大容量款调研结论：**'White' (白色)** 是最大的机会点。大量用户抱怨白色先用完，导致套装闲置。建议：套装内配置双倍容量白色。")

st.write("---")
st.write("📋 原始评论抽样 (前 20 条):")
st.dataframe(selected_df[['body', 'data_type']].head(20))
