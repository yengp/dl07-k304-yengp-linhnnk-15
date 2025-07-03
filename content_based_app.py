import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


st.set_page_config(layout="wide")
st.title("Tổng quan đánh giá công ty từ ITViec")

# Đọc dữ liệu
df = pd.read_excel('Processed_reviews.xlsx')

tab1, tab2 = st.tabs(["🔍 Tìm review theo từ khóa", "🏢 Tổng quan công ty"])

with tab1:
    st.header("1. Tìm công ty theo từ khóa nổi bật")
    keyword = st.text_input("Nhập từ khóa bạn quan tâm:")
    if keyword:
        temp_df = df.copy()
        # Lọc các review chứa từ khóa ở 1 trong 2 cột
        mask = (
            temp_df["What I liked"].fillna("").str.contains(keyword, case=False, na=False) |
            temp_df["Suggestions for improvement"].fillna("").str.contains(keyword, case=False, na=False)
        )
        filtered = temp_df.loc[mask, ["Company Name", "What I liked", "Suggestions for improvement", "sentiment"]].dropna(how='all').head(10)
        if not filtered.empty:
            st.dataframe(filtered.rename(columns={
                "Company Name": "Tên công ty",
                "What I liked": "Điều tôi thích về công ty này",
                "Suggestions for improvement": "Góp ý khắc phục",
                "sentiment": "Loại cảm xúc"
            }))
        else:
            st.info("Không tìm thấy công ty có review phù hợp với từ khóa.")

with tab2:
    st.header("2. Tổng quan & trực quan hóa công ty")
    company2 = st.selectbox("Chọn công ty", df["Company Name"].dropna().unique(), key="company2")
    company_df = df[df["Company Name"] == company2]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tổng quan")
        st.write(f"Số lượt đánh giá: **{len(company_df)}**")
        st.write(f"Điểm trung bình: **{company_df['Rating'].mean():.2f}**")
        st.write("**Phân bố cảm xúc:**")
        sentiment_counts = company_df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

        st.write("**Nhận xét tích cực nổi bật:**")
        st.write(company_df[company_df["sentiment"] == "positive"]["What I liked"].dropna().head(3).tolist())
        st.write("**Nhận xét tiêu cực nổi bật:**")
        st.write(company_df[company_df["sentiment"] == "negative"]["Suggestions for improvement"].dropna().head(3).tolist())

    with col2:
        st.subheader("WordCloud tích cực")
        pos_text = " ".join(company_df[company_df['sentiment'] == 'positive']['liked_clean'].dropna())
        if pos_text:
            wc = WordCloud(width=400, height=200, background_color="white").generate(pos_text)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        st.subheader("WordCloud tiêu cực")
        neg_text = " ".join(company_df[company_df['sentiment'] == 'negative']['suggestion_clean'].dropna())
        if neg_text:
            wc = WordCloud(width=400, height=200, background_color="white").generate(neg_text)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

    # Cụm phù hợp (nếu có dữ liệu)
    if {'x', 'y', 'cluster'}.issubset(company_df.columns):
        st.subheader("Phân cụm đánh giá (KMeans)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=company_df, x="x", y="y", hue="cluster", palette="Set2", ax=ax)
        st.pyplot(fig)
