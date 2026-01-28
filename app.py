import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. 页面基本配置
st.set_page_config(page_title="丙烯市场调研看板", layout="wide")
st.title("🎨 丙烯颜料评论分析：儿童款 vs 大容量款")

# 2. 数据整合逻辑 (直接写在主程序中)
@st.cache_data
def load_combined_data():
    # 严格匹配你上传的文件名
    files = {
        "kids_sales.xlsx - Sheet1.csv": ("儿童", "销量Top10"),
        "kids_trending.xlsx - Sheet1.csv": ("儿童", "趋势Top10"),
        "large_capacity_sales.xlsx - Sheet2.csv": ("大容量", "销量Top10"),
        "large_capacity_trending.xlsx - Sheet1.csv": ("大容量", "趋势Top10")
    }
    
    combined = []
    for filename, info in files.items():
        try:
            # 尝试读取数据
            df = pd.read_csv(filename)
            df['category'] = info[0]  # 儿童 或 大容量
            df['data_type'] = info[1] # 销量 或 趋势
            # 确保评论列存在且为字符串
            if 'body' in df.columns:
                df['body'] = df['body'].fillna('').astype(str)
                combined.append(df)
        except Exception as e:
            st.warning(f"文件 {filename} 读取跳过。错误原因: {e}")
            
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

# 加载数据
df = load_combined_data()

# 3. 侧边栏与数据筛选
st.sidebar.header("数据筛选")
selected_cat = st.sidebar.radio("选择调研产品线", ["儿童丙烯", "大容量定位丙烯"])
# 提取关键词进行过滤
filter_tag = "儿童" if "儿童" in selected_cat else "大容量"
selected_df = df[df['category'] == filter_tag]

# 4. 分析逻辑：定义关键词库
# 针对大容量款增加了“白色不够用”的相关特征词
pain_points = {
    "白色不够/消耗快": ["white", "more white", "ran out", "empty white", "extra white"],
    "笔尖干涸/堵塞": ["dry", "dried", "clog", "stuck", "fast drying"],
    "漏液/包装差": ["leak", "mess", "spilled", "broken", "seal"],
    "覆盖力/质地": ["sheer", "thin", "watery", "transparent"]
}

highlights = {
    "收纳/便携性": ["case", "box", "storage", "organizer", "carrying"],
    "色彩鲜艳": ["vibrant", "bright", "colors", "pigment", "rich"],
    "礼品属性": ["gift", "present", "granddaughter", "son", "kid"],
    "性价比/大容量": ["value", "deal", "ounce", "large", "volume", "affordable"]
}

# 辅助函数：统计关键词
def analyze_text(data, kw_dict):
    results = {}
    for label, keywords in kw_dict.items():
        # 使用正则表达式匹配多个词，不区分大小写
        pattern = '|'.join(keywords)
        results[label] = data['body'].str.contains(pattern, case=False, na=False).sum()
    return pd.Series(results).sort_values(ascending=False)

# 5. 页面展示布局
if not selected_df.empty:
    tab1, tab2, tab3 = st.tabs(["📊 满意点与痛点", "👥 用户画像", "💡 行动建议"])

    with tab1:
        st.subheader(f"🔍 {selected_cat} - 核心评价分布")
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ 满意点 (High Lights)")
            hi_counts = analyze_text(selected_df, highlights)
            st.bar_chart(hi_counts)
        with col2:
            st.error("❌ 不满意点 (Pain Points)")
            pain_counts = analyze_text(selected_df, pain_points)
            st.bar_chart(pain_counts)

    with tab2:
        st.subheader("👤 目标买家画像 (基于关键词匹配)")
        persona_kw = {
            "家长/送礼群体": ["gift", "kid", "child", "grand", "school"],
            "画师/DIY博主": ["artist", "mural", "canvas", "professional", "rock painting"]
        }
        persona_counts = analyze_text(selected_df, persona_kw)
        fig = px.pie(values=persona_counts.values, names=persona_counts.index, hole=.4)
        st.plotly_chart(fig)

    with tab3:
        st.subheader("🚀 调研总结与行动建议")
        if filter_tag == "儿童":
            st.info("""
            **结论**：儿童款受‘收纳盒’和‘送礼’驱动明显。
            **建议**：我们的产品应标配精美收纳盒；主图增加‘送礼场景’；确保盖子易拉开（避免儿童吐槽）。
            """)
        else:
            st.info("""
            **结论**：大容量款用户极度关注‘覆盖力’和‘白色颜料余量’。
            **建议**：产品配置中**显著加大白色的毫升数**；主打‘画墙不透底’卖点；优化密封性防止大瓶干涸。
            """)
else:
    st.error("未找到相关数据，请检查文件名是否正确上传至仓库。")
