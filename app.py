import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import plotly.express as px
import matplotlib.pyplot as plt

# 1. 页面配置
st.set_page_config(page_title="丙烯市场调研报告", layout="wide")
st.title("🎨 丙烯颜料评论分析看板")

# 2. 加载数据 (根据你的文件名调整)
@st.cache_data
def load_all_data():
    files = {
        "儿童款-销量Top10": "kids_sales.xlsx - Sheet1.csv",
        "儿童款-趋势Top10": "kids_trending.xlsx - Sheet1.csv",
        "成人款-销量Top10": "large_capacity_sales.xlsx - Sheet2.csv",
        "成人款-趋势Top10": "large_capacity_trending.xlsx - Sheet1.csv"
    }
    combined = []
    for name, path in files.items():
        try:
            df = pd.read_csv(path)
            df['source'] = name
            df['category'] = "儿童" if "儿童" in name else "成人"
            combined.append(df)
        except:
            continue
    return pd.concat(combined) if combined else pd.DataFrame()

df = load_all_data()

# 3. 侧边栏导航
st.sidebar.header("筛选条件")
category = st.sidebar.radio("选择产品线", ["儿童丙烯", "成人大容量丙烯"])
selected_df = df[df['category'] == category[:2]]

# 4. 核心逻辑：定义关键词
pain_points = {
    "白色不够": ["white", "more white", "ran out of white", "extra white"],
    "干燥问题": ["dry", "dried up", "fast drying", "clogged"],
    "包装/漏液": ["leak", "mess", "spilled", "broken"],
    "覆盖力差": ["sheer", "thin", "coverage", "watery"]
}

highlights = {
    "收纳好评": ["case", "box", "organizer", "storage"],
    "颜色丰富": ["vibrant", "colors", "pigment", "bright"],
    "送礼推荐": ["gift", "present", "daughter", "son", "grandkids"],
    "性价比高": ["value", "cheap", "price", "affordable"]
}

# 5. 展示报告内容
tab1, tab2, tab3 = st.tabs(["📊 满意度与痛点", "👥 用户画像", "💡 行动建议"])

with tab1:
    st.subheader(f"{category} 核心评论特征")
    col1, col2 = st.columns(2)
    
    # 简单的关键词提取逻辑
    def count_keywords(data, kw_dict):
        results = {}
        for label, keywords in kw_dict.items():
            count = data['body'].str.contains('|'.join(keywords), case=False, na=False).sum()
            results[label] = count
        return pd.Series(results).sort_values(ascending=False)

    with col1:
        st.success("✅ 满意点统计")
        hi_counts = count_keywords(selected_df, highlights)
        st.bar_chart(hi_counts)

    with col2:
        st.error("❌ 不满意点统计")
        pain_counts = count_keywords(selected_df, pain_points)
        st.bar_chart(pain_counts)

with tab2:
    st.subheader("谁在购买？")
    # 通过关键词判断画像
    persona_kw = {"家长/送礼": ["gift", "kid", "child", "son", "daughter"], "画师/博主": ["professional", "mural", "canvas", "artist"]}
    persona_counts = count_keywords(selected_df, persona_kw)
    fig = px.pie(values=persona_counts.values, names=persona_counts.index, hole=.3)
    st.plotly_chart(fig)

with tab3:
    st.subheader("📢 市场行动建议")
    if "儿童" in category:
        st.markdown("""
        - **主打卖点**：强化“收纳盒”优势，突出“礼品属性”。
        - **改进方向**：检查盖子是否易于儿童开启，增加更多亮色系。
        """)
    else:
        st.markdown("""
        - **主打卖点**：强调“大容量”和“覆盖力”。
        - **核心痛点**：**必须增加白色颜料的配比**，或提供单独的白色替换装。
        - **改进方向**：优化瓶口设计防止干涸。
        """)
