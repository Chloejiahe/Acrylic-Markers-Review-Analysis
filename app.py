import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="丙烯调研报告", layout="wide")
st.title("🎨 丙烯颜料市场调研分析看板")

# --- 1. 动态数据加载逻辑 ---
@st.cache_data
def load_data_robust():
    # 获取当前目录下所有文件
    all_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))
    
    combined = []
    # 定义匹配逻辑
    for path in all_files:
        p_lower = path.lower()
        cat, dtype = None, None
        
        if "kids" in p_lower: cat = "儿童"
        elif "large" in p_lower or "capacity" in p_lower: cat = "大容量"
        
        if "sales" in p_lower: dtype = "销量Top10"
        elif "trending" in p_lower: dtype = "趋势Top10"
        
        if cat and dtype:
            try:
                # 尝试读取，加上 encoding 处理可能存在的特殊字符
                tmp = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
                tmp['category'] = cat
                tmp['data_type'] = dtype
                
                # 关键修复：将你的 'Content' 列重命名为代码通用的 'body'
                if 'Content' in tmp.columns:
                    tmp = tmp.rename(columns={'Content': 'body'})
                elif 'English Content' in tmp.columns: # 备选列名
                    tmp = tmp.rename(columns={'English Content': 'body'})
                
                combined.append(tmp)
                st.sidebar.success(f"已加载: {os.path.basename(path)}")
            except Exception as e:
                st.sidebar.error(f"读取失败 {path}: {e}")

    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

df = load_data_robust()

# --- 2. 报错诊断 ---
if df.empty or 'category' not in df.columns:
    st.error("🚨 数据加载失败！请检查以下事项：")
    st.write("1. 确认 CSV 文件已上传到 GitHub 仓库根目录。")
    st.write("2. 当前检测到的文件列表：", os.listdir('.'))
    st.stop()

# --- 3. 筛选器 ---
st.sidebar.divider()
choice = st.sidebar.radio("选择调研产品线", ["儿童丙烯", "大容量款"])
target = "儿童" if "儿童" in choice else "大容量"
selected_df = df[df['category'] == target].copy()

# --- 4. 调研报告核心内容 ---
st.header(f"📊 {choice} 调研发现")

tab1, tab2, tab3 = st.tabs(["💡 满意点与痛点", "👥 用户画像", "📢 行动建议"])

with tab1:
    col1, col2 = st.columns(2)
    # 针对你上传的数据内容优化关键词
    high_kws = {"色彩/覆盖力": "vibrant|bright|coverage|opacity|pigment", "收纳设计": "case|box|storage|organized", "易用性": "easy|flow|smooth|marker"}
    pain_kws = {"白色缺失": "white|ran out|more white|extra white", "干涸堵塞": "dry|dried|stuck|clog", "包装漏液": "leak|mess|spilled|broken"}

    def get_counts(data, kw_dict):
        # 确保 body 列是字符串
        data['body'] = data['body'].fillna('').astype(str)
        return pd.Series({k: data['body'].str.contains(v, case=False, na=False).sum() for k, v in kw_dict.items()})

    with col1:
        st.success("✅ 满意点排行")
        st.bar_chart(get_counts(selected_df, high_kws))
    with col2:
        st.error("❌ 痛点排行")
        st.bar_chart(get_counts(selected_df, pain_kws))

with tab2:
    st.subheader("谁在购买？")
    persona_kws = {"家长/送礼 (Gift/Grandkid)": "gift|grand|child|son|daughter", "专业/画师 (Artist)": "artist|professional|mural|canvas|rock"}
    p_counts = get_counts(selected_df, persona_kws)
    st.plotly_chart(px.pie(values=p_counts.values, names=p_counts.index, hole=0.4))

with tab3:
    st.subheader("市场行动建议 (老板参考)")
    if target == "儿童":
        st.info("🎯 **核心发现**：儿童款用户对 **'Case' (收纳盒)** 的依赖度极高，常作为礼物（Grandkids/Gift）。\n\n✅ **建议**：强化提手收纳盒设计，主打礼品包装。")
    else:
        st.warning("🎯 **核心发现**：大容量款用户（画墙/石头画）对 **'White' (白色)** 的消耗速度远超预期，白色干涸是第二大痛点。\n\n✅ **建议**：套装内增加一支备用白色，或在详情页强调白色大容量。")

# 5. 原始数据查看
with st.expander("查看原始评论"):
    st.dataframe(selected_df[['body', 'data_type']].head(50))
