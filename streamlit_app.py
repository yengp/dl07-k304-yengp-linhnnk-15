import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from predictor import predict_sentiment, load_model_components
import datetime

st.set_page_config(layout="wide")
st.title("Hệ thống Phân tích & Đề xuất Doanh nghiệp IT")

# --- CSS để cố định thông tin lớp/học viên ở cuối sidebar ---
st.markdown("""
<style>
    .st-emotion-cache-vk3305 { /* Đây là class của sidebar chính */
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Đẩy nội dung lên trên và xuống dưới */
    }
    .fixed-bottom-left {
        position: sticky; /* Hoặc fixed nếu bạn muốn nó luôn ở đó ngay cả khi cuộn */
        bottom: 0;
        left: 0; /* Đảm bảo nó nằm sát mép trái của sidebar */
        width: 100%; /* Chiếm toàn bộ chiều rộng của sidebar */
        padding: 1rem; /* Thêm padding cho đẹp */
        background-color: #f0f2f6; /* Màu nền giống sidebar hoặc màu bạn muốn */
        border-top: 1px solid #e0e0e0; /* Đường viền phía trên để tách biệt */
        box-sizing: border-box; /* Đảm bảo padding không làm tăng kích thước */
        font-size: 0.95rem;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Logo, Menu và Thông tin học viên ---
with st.sidebar:
    st.image("img/your_logo.jpg", width=60)
    st.markdown("<h3 style='margin-bottom:0.5rem;'>IT Recommendation System</h3>", unsafe_allow_html=True)

    menu_options = {
        "Business Problem": "💼 Phân tích nghiệp vụ",
        "Build Project": "🛠️ Xây dựng mô hình",
        "New Prediction": "📊 Phân tích công ty"
    }
    menu_labels = list(menu_options.values())
    menu_keys = list(menu_options.keys())

    # Không cần index, chỉ lấy mặc định là 0
    selected_label = st.selectbox(
        "Menu",
        menu_labels,
        index=0
    )
    # Lấy key thực tế từ label
    menu_selection = menu_keys[menu_labels.index(selected_label)]

    # Highlight mục đang chọn (CSS)
    st.markdown("""
    <style>
    .stSelectbox [data-baseweb="select"] > div {
        font-weight: bold;
        background: #e6f0ff;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="fixed-bottom-left" style="background-color:#f0f2f6; border-top:1px solid #e0e0e0;">
        <b>LỚP DL07_K304 - DATA SCIENCE - MACHINE LEARNING</b><br>
        Học viên thực hiện:<br>
        - <b>Ms. Giang Phi Yến</b> - <a href='mailto:yengp96@gmail.com'>Email</a><br>
        - <b>Ms. Nguyễn Ngọc Khánh Linh</b> - <a href='mailto:nnkl1517000@gmail.com'>Email</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- 1. Business Problem ---
if menu_selection == "Business Problem":
    st.header("Hiểu rõ Vấn đề Kinh doanh")
    st.markdown("""
    Ứng dụng này nhằm giải quyết hai vấn đề cốt lõi trong lĩnh vực tuyển dụng và đánh giá doanh nghiệp IT tại Việt Nam.
    Chúng tôi sử dụng dữ liệu từ nền tảng tuyển dụng **ITviec.com** để cung cấp các phân tích và dự đoán hữu ích nhằm hỗ trợ công ty và người lao động.
    """)

    st.subheader("1.1. Phân tích Cảm xúc từ Review (Sentiment Analysis)")
    st.markdown("""
    * **Yêu cầu:** Phân tích các đánh giá (review) được đăng bởi ứng viên hoặc nhân viên về các công ty trên nền tảng **ITviec**.
    * **Nguồn dữ liệu:** Bao gồm các trường như nội dung tích cực, góp ý cải thiện, điểm đánh giá...
    * **Mục tiêu:** Dự đoán cảm xúc tương ứng với từng review (tích cực / tiêu cực / trung tính). Kết quả này hỗ trợ các công ty:
        - Theo dõi phản hồi từ nhân viên/ứng viên.
        - Phản ứng nhanh với các vấn đề nội bộ.
        - Cải thiện hình ảnh thương hiệu nhà tuyển dụng.
    """)
    st.info("💡 Bạn có thể trải nghiệm phân tích này trong phần 'Build Project' và dự đoán nhanh tại 'New Prediction'.")

    st.subheader("1.2. Phân Cụm Thông Tin Đánh Giá (Information Clustering)")
    st.markdown("""
    * **Yêu cầu:** Dựa trên nội dung review để phân loại nhóm đánh giá mà công ty đang thuộc về.
    * **Nguồn dữ liệu:** Văn bản đánh giá từ nhiều công ty trên ITviec.
    * **Mục tiêu:** Giúp công ty hiểu được bản thân đang nằm trong nhóm nào (ví dụ: nhóm bị chê quản lý – nhóm nổi bật về đào tạo – nhóm có chính sách tốt...).
        - So sánh với đối thủ cùng ngành.
        - Xác định nhóm điểm mạnh và yếu để ưu tiên cải thiện.
    """)
    st.info("💡 Bạn có thể xem cụ thể từng nhóm/cụm phân tích trong phần 'Build Project'.")

# --- 2. Build Project ---
elif menu_selection == "Build Project":
    st.header("2. Build Project (Data Science Pipeline)")
    tabs_build = st.tabs([
        "Business Understanding",
        "Data Understanding",
        "Data Preparation",
        "Modeling",
        "Evaluation",
        "Deployment"
    ])

    with tabs_build[0]:
        st.subheader("Bước 1: Hiểu bài toán (Business Understanding)")
        st.markdown("""
        - **Mục tiêu 1:** Dự đoán cảm xúc review.
        - **Mục tiêu 2:** Phân cụm nội dung đánh giá.
        """)

    with tabs_build[1]:
        st.subheader("Bước 2: Khám phá và chọn lọc dữ liệu (Data Understanding)")
        st.markdown("""
        - Cho phép chọn công ty cụ thể để phân tích.
        - Hiển thị số lượng review, thời gian, điểm đánh giá, v.v.
        """)
        st.markdown("""
        ---
        **Dữ liệu đầu vào sử dụng cho phân tích gồm 3 tệp chính:**

        1. **Overview_Companies.xlsx**
            - Thông tin tổng quan về các công ty.

        2. **Overview_Reviews.xlsx**
            - Thống kê điểm đánh giá tổng quan của mỗi công ty.
            - *Mục đích sử dụng:* Phân tích tổng hợp công ty, đánh giá tổng thể theo chiều rộng (xác định công ty nào tốt/xấu).

        3. **Reviews.xlsx**
            - Nội dung review chi tiết từ người dùng.
            - *Đặc điểm:* Trích xuất cảm xúc (từ các phần như "What I liked", "Suggestions"...).
            - *Mục đích sử dụng:* Dùng cho mô hình phân cụm (clustering), LDA (Latent Dirichlet Allocation), wordcloud.
            - *Vai trò:* Là nguồn đầu vào chính cho hầu hết các phân tích.
        """)

    with tabs_build[2]:
        st.subheader("Bước 3: Chuẩn bị dữ liệu (Data Preparation)")
        st.markdown("""
        - Tiền xử lý văn bản: xóa stopwords, chuẩn hóa tiếng Việt, tokenize (sử dụng Underthesea).
        - TF-IDF hoặc CountVectorizer cho feature extraction.
        """)

    with tabs_build[3]:
        st.subheader("Bước 4: Modeling")
        st.markdown("""
        - **Sentiment Analysis:** XGBoost/ Logistic Regression / RandomForest / SVM / Naive Bayes/ K-Nearest Neighbors  
        → *Mô hình XGBoost là mô hình có hiệu suất cao nhất về tất cả các tiêu chí.*
        - **Clustering:** KMeans / Agglomerative / DBSCAN + LDA (gợi ý số cụm).  
        → *Mô hình LDA + KMeans tốt nhất - Hiệu quả nhất, trực quan đẹp, dễ diễn giải.*
        """)

    with tabs_build[4]:
        st.subheader("Bước 5: Evaluation")
        st.markdown("""
        - Đánh giá phân loại: accuracy, precision, recall, F1, confusion matrix, ROC curve.
        - Đánh giá clustering: Silhouette score, biểu đồ tường minh cụm.
        """)

    with tabs_build[5]:
        st.subheader("Bước 6: Deployment")
        st.markdown("""
        - Hiển thị dashboard phân tích cho từng công ty:
            - Wordcloud tích cực / tiêu cực
            - Từ khóa nổi bật
            - Cụm thông tin mà công ty thuộc về
            - Gợi ý cải thiện
        """)

# --- 3. New Prediction ---
elif menu_selection == "New Prediction":
    st.header("📊 Phân tích Công ty Cụ thể từ Review")

    # 1. Chọn công ty
    with st.spinner("Đang tải dữ liệu..."):
        df = pd.read_excel("Processed_reviews.xlsx")

    # Bộ lọc thời gian nếu có cột ngày
    if "Review Date" in df.columns:
        df["Review Date"] = pd.to_datetime(df["Review Date"])
        min_date = df["Review Date"].min()
        max_date = df["Review Date"].max()
        date_range = st.slider(
            "Chọn khoảng thời gian",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="DD/MM/YYYY"
        )
        with st.spinner("Đang lọc dữ liệu theo thời gian..."):
            df = df[(df["Review Date"] >= date_range[0]) & (df["Review Date"] <= date_range[1])]

    company = st.selectbox("Chọn công ty để phân tích", df["Company Name"].dropna().unique())
    company_df = df[df["Company Name"] == company]

    # 2. Thống kê cơ bản
    st.markdown("### 1️⃣ Tổng quan đánh giá")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tổng số review", len(company_df))
        st.metric("Điểm đánh giá trung bình", f"{company_df['Rating'].mean():.2f}")
    with col2:
        st.write("#### Phân bố cảm xúc:")
        sentiment_counts = company_df["sentiment"].value_counts()
        sentiment_perc = sentiment_counts / sentiment_counts.sum() * 100
        # Vẽ biểu đồ với màu phù hợp
        fig, ax = plt.subplots()
        colors = ["#2b83ba", "#fdae61", "#cccccc"]  # xanh dương, cam, xám
        sentiment_order = ["positive", "negative", "neutral"]
        plot_counts = [sentiment_counts.get(s, 0) for s in sentiment_order]
        ax.bar(sentiment_order, plot_counts, color=colors)
        ax.set_ylabel("Số lượng review")
        ax.set_title("Phân bố cảm xúc")
        st.pyplot(fig)

    # 3. Nhận xét nổi bật
    st.markdown("### 2️⃣ Nhận xét tiêu biểu")

    # Chọn cảm xúc để xem chi tiết review
    sentiment_labels = []
    sentiment_counts = company_df["sentiment"].value_counts()
    sentiment_perc = sentiment_counts / sentiment_counts.sum() * 100
    for sentiment in sentiment_counts.index:
        count = sentiment_counts[sentiment]
        perc = sentiment_perc[sentiment]
        sentiment_labels.append(f"{sentiment} ({count} reviews, {perc:.1f}%)")
    sentiment_map = dict(zip(sentiment_labels, sentiment_counts.index))
    selected_sentiment = st.radio(
        "Chọn cảm xúc để xem chi tiết review:",
        sentiment_labels,
        horizontal=True
    )
    chosen_sentiment = sentiment_map[selected_sentiment]

    # Bảng kết quả review theo cảm xúc đã chọn
    st.markdown(f"**Danh sách review với cảm xúc: _{chosen_sentiment}_**")
    review_cols = ["What I liked", "Suggestions for improvement", "Rating"]
    if "Review Date" in company_df.columns:
        review_cols = ["Review Date"] + review_cols
    st.dataframe(
        company_df[company_df["sentiment"] == chosen_sentiment][review_cols].reset_index(drop=True)
    )

    # 4. WordCloud
    st.markdown("### 3️⃣ WordCloud cảm xúc")
    col3, col4 = st.columns(2)
    with col3:
        st.write("**Tích cực**")
        pos_text = " ".join(company_df[company_df["sentiment"] == "positive"]["liked_clean"].dropna())
        if pos_text:
            wc_pos = WordCloud(width=400, height=200, background_color="white").generate(pos_text)
            fig, ax = plt.subplots()
            ax.imshow(wc_pos, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
    with col4:
        st.write("**Tiêu cực**")
        neg_text = " ".join(company_df[company_df["sentiment"] == "negative"]["suggestion_clean"].dropna())
        if neg_text:
            wc_neg = WordCloud(width=400, height=200, background_color="white").generate(neg_text)
            fig, ax = plt.subplots()
            ax.imshow(wc_neg, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

    # 5. Phân cụm
    if {"x", "y", "cluster"}.issubset(company_df.columns):
        st.markdown("### 4️⃣ Phân cụm thông tin đánh giá (KMeans)")
        fig, ax = plt.subplots()
        sns.scatterplot(data=company_df, x="x", y="y", hue="cluster", palette="Set2", ax=ax)
        ax.set_title("Biểu đồ phân cụm review")
        st.pyplot(fig)

        st.markdown("**Thông tin theo cụm**:")
        for cluster_id in sorted(company_df["cluster"].unique()):
            st.markdown(f"#### 🔸 Cụm {cluster_id}")
            cluster_reviews = company_df[company_df["cluster"] == cluster_id]
            st.write("**Từ khóa chính:**")
            keyword_cluster = " ".join(cluster_reviews["liked_clean"].fillna("") + " " + cluster_reviews["suggestion_clean"].fillna(""))
            if keyword_cluster:
                wordcloud = WordCloud(width=600, height=200).generate(keyword_cluster)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            st.caption(f"Tổng số review trong cụm: {len(cluster_reviews)}")

    # 6. Đề xuất tổng hợp
    st.markdown("### 5️⃣ Đề xuất cải thiện cho công ty")
    if company_df["sentiment"].value_counts().get("negative", 0) > company_df["sentiment"].value_counts().get("positive", 0):
        st.warning("Công ty đang nhận nhiều review tiêu cực hơn tích cực. Nên tập trung cải thiện các vấn đề sau:")
    else:
        st.success("Công ty nhận nhiều đánh giá tích cực. Tuy nhiên vẫn có thể cải thiện thêm những điểm sau:")

    suggestions = company_df["Suggestions for improvement"].dropna().head(5).tolist()
    for s in suggestions:
        st.markdown(f"- {s}")
