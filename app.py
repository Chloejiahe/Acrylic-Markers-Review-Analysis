import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. 页面配置
st.set_page_config(page_title="丙烯调研报告", layout="wide")
st.title("🎨 丙烯颜料市场调研看板")

# 2. 自动扫描并读取数据
@st.cache_data
def auto_load_data():
    all_files = os.listdir('.')  # 扫描根目录所有文件
    combined = []
    
    # 逻辑映射：关键字 -> (分类, 类型)
    mapping = {
        ("kids", "sales"): ("儿童", "销量Top10"),
        ("kids", "trending"): ("儿童", "趋势Top10"),
        ("large", "sales"): ("大容量", "销量Top10"),
        ("large", "trending"): ("大容量", "趋势Top10")
    }
    
    for filename in all_files:
        fname_lower = filename.lower()
        for keywords, info in mapping.items():
            # 只要文件名里包含 kids 和 sales 等关键字，就尝试读取
            if all(k in fname_lower for k in keywords) and filename.endswith('.csv'):
                try:
                    df = pd.read_csv(filename)
                    df['category'] = info[0]
                    df['data_type'] = info[1]
                    # 根据你上传的文件内容，评论列其实叫 'Content'
                    if 'Content' in df.columns:
                        df = df.rename(columns={'Content': 'body'})
                    combined.append(df)
                    st.sidebar.success(f"✅ 已识别并加载: {filename}")
                except Exception as e:
                    st.sidebar.error(f"读取 {filename} 失败: {e}")
                    
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

df = auto_load_data()

# 3. 筛选与分析
st.sidebar.divider()
choice = st.sidebar.radio("选择产品线", ["儿童丙烯", "大容量款"])
target_tag = "儿童" if "儿童" in choice else "大容量"
selected_df = df[df['category'] == target_tag]

# 4. 老板要求的调研维度分析
st.header(f"📊 {choice} 深度调研报告")

tab1, tab2, tab3 = st.tabs(["核心特质分析", "用户画像", "行动建议"])

with tab1:
    col1, col2 = st.columns(2)
    # 定义匹配词库
    highlights = {"收纳盒好评": "case|box|storage|organized", "色彩鲜艳": "vibrant|bright|pigment", "易于使用": "easy|smooth|flow"}
    pains = {"白色不够用": "white|ran out|more white|empty", "容易干涸": "dry|dried|stuck|clog", "覆盖力差": "coverage|thin|watery|transparent"}
    
    def count_kws(data, d):
        return pd.Series({k: data['body'].str.contains(v, case=False, na=False).sum() for k, v in d.items()})

    with col1:
        st.success("✅ 满意点 (Highlights)")
        st.bar_chart(count_kws(selected_df, highlights))
    with col2:
        st.error("❌ 不满意点 (Pain Points)")
        st.bar_chart(count_kws(selected_df, pains))

with tab2:
    st.subheader("👥 谁在买？")
    persona = {"家长送礼": "gift|grandchild|son|daughter|kids", "博主/画师": "artist|professional|mural|canvas|rock"}
    p_counts = count_kws(selected_df, persona)
    st.plotly_chart(px.pie(values=p_counts.values, names=p_counts.index, hole=0.4))

with tab3:
    st.subheader("💡 调研行动建议")
    if target_tag == "儿童":
        st.info("儿童款核心：用户极其看重 **Case (收纳盒)**。建议新品加强包装的耐用性，主打礼品属性。")
    else:
        st.warning("大容量款核心：**白色颜料是个大坑**。评论频繁反馈白色用完。建议：套装内增加白色比例，或赠送两支白色。")

st.write("---")
st.write("📋 原始评论抽样：")
st.dataframe(selected_df[['body', 'data_type']].head(10))
