import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re
from underthesea import word_tokenize
from wordcloud import WordCloud

st.set_page_config(page_title="Phân tích & Phân cụm Review ITViec", layout="wide")

# Load dữ liệu mẫu
@st.cache_data
def load_data():
    reviews = pd.read_excel("Reviews.xlsx")
    overview = pd.read_excel("Overview_Companies.xlsx")
    columns_to_add = ['id', 'Company Type', 'Company industry', 'Company size', 'Country', 'Working days', 'Overtime Policy']
    reviews = reviews.merge(overview[columns_to_add], on='id', how='left')
    reviews['combined_text'] = (
        reviews['Title'].fillna('') + ' ' +
        reviews['What I liked'].fillna('') + ' ' +
        reviews['Suggestions for improvement'].fillna('')
    )
    return reviews

# Tiền xử lý văn bản
@st.cache_data
def preprocess_text(text):
    text = re.sub(r"[\n\r]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = word_tokenize(text, format="text")
    return text

# Load dữ liệu
Reviews = load_data()
Reviews['text_processed'] = Reviews['combined_text'].apply(preprocess_text)

# Danh sách công ty
company_list = Reviews['Company Name'].dropna().unique().tolist()
company_list.sort()

# === MENU DROPDOWN ===
menu = st.sidebar.selectbox("Menu", ["Thông tin đồ án", "Thông tin công ty", "Phân tích cảm xúc", "Phân cụm đánh giá"])

if menu == "Thông tin đồ án":
    st.markdown("""
    <h1 style='font-size:36px;'>📁 THÔNG TIN ĐỒ ÁN</h1>
    <div style='font-size:22px; line-height:1.8;'>
        <b>Tên đồ án:</b> <span style='color:#00CED1; font-size:24px;'>Sentiment Analysis and Information Clustering</span><br>
        <b>Người thực hiện:</b> <span style='color:#FFD700; font-size:20px;'>Lê Hữu Sơn Hải</span> --- <span style='color:#ADFF2F; font-size:20px;'>Đoàn Trung Cường</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 📝 Mô tả đồ án
    Dự án nhằm phân tích các đánh giá từ ứng viên hoặc nhân viên về các công ty trên nền tảng ITViec. 
    Dựa vào các đánh giá này, hệ thống sẽ thực hiện:
    - **Phân tích cảm xúc**: Dự đoán thái độ tích cực, tiêu cực hay trung lập.
    - **Phân cụm đánh giá**: Nhóm các đánh giá có nội dung tương đồng để hỗ trợ công ty cải thiện môi trường làm việc.
    """)

    st.markdown("### 📊 Biểu đồ thống kê ngành nghề phổ biến")
    industry_counts = Reviews['Company industry'].value_counts()
    fig_industry, ax_industry = plt.subplots(figsize=(8, 6))
    sns.barplot(x=industry_counts.values, y=industry_counts.index, ax=ax_industry)
    ax_industry.set_xlabel("Số công ty", fontsize=12)
    ax_industry.set_ylabel("Ngành nghề", fontsize=12)
    st.pyplot(fig_industry)

elif menu == "Thông tin công ty":
    st.title("🏢 THÔNG TIN CÔNG TY")
    selected_company_info = st.selectbox("Chọn công ty muốn xem thông tin:", ["" ] + company_list)
    if selected_company_info:
        company_info_data = Reviews[Reviews['Company Name'] == selected_company_info]
        if not company_info_data.empty:
            st.markdown(f"### Kết quả cho công ty: **{selected_company_info}**")
            company_overview = company_info_data[[
                'Company Name', 'Company Type', 'Company industry', 'Company size', 'Country', 'Working days', 'Overtime Policy'
            ]].drop_duplicates()
            st.dataframe(company_overview)
            st.markdown("### Các đánh giá liên quan")
            st.dataframe(company_info_data[['combined_text']])
        else:
            st.warning("Không tìm thấy công ty phù hợp.")

elif menu == "Phân tích cảm xúc":
    st.title("📊 Phân tích cảm xúc từ Review ứng viên")
    st.markdown("Phân loại cảm xúc thành tích cực, tiêu cực hoặc trung tính.")

    # Hiển thị danh sách công ty thiếu dữ liệu
    with st.expander("📌 Xem danh sách công ty thiếu dữ liệu cảm xúc"):
        no_rating = Reviews[Reviews['Rating'].isna()]['Company Name'].dropna().unique().tolist()
        no_review = Reviews[Reviews['combined_text'].str.strip() == '']['Company Name'].dropna().unique().tolist()

        if not no_rating and not no_review:
            st.success("Tất cả các công ty đều có dữ liệu đầy đủ.")
        else:
            if no_rating:
                st.warning("❗ Công ty không có Rating (không phân tích cảm xúc được):")
                for comp in no_rating:
                    st.markdown(f"- {comp}")
            if no_review:
                st.warning("❗ Công ty không có Review (không phân tích được):")
                for comp in no_review:
                    st.markdown(f"- {comp}")

    selected_company = st.selectbox("Chọn công ty để phân tích cảm xúc:", company_list)
    company_data = Reviews[Reviews['Company Name'] == selected_company].copy()

    if 'Sentiment' not in company_data.columns:
        if 'Rating' in company_data.columns:
            def map_rating_to_sentiment(r):
                if r >= 4:
                    return "Tích cực"
                elif r == 3:
                    return "Trung tính"
                else:
                    return "Tiêu cực"
            company_data['Sentiment'] = company_data['Rating'].apply(map_rating_to_sentiment)
        else:
            st.warning("Không có cột nhãn cảm xúc hoặc đánh giá để train mô hình.")

    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X = tfidf_vectorizer.fit_transform(company_data['text_processed'])
    y = company_data['Sentiment']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    company_data['Predicted Sentiment'] = y_pred

    st.subheader("Kết quả phân tích cảm xúc")
    st.dataframe(company_data[['combined_text', 'Sentiment', 'Predicted Sentiment']])

    st.subheader("Biểu đồ phân bố cảm xúc")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Predicted Sentiment', data=company_data, ax=ax)
    st.pyplot(fig)

    st.subheader("Báo cáo phân loại")
    st.text(classification_report(y, y_pred))

    st.subheader("🔍 Ma trận nhầm lẫn")
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax_cm)
    ax_cm.set_xlabel("Dự đoán")
    ax_cm.set_ylabel("Thực tế")
    st.pyplot(fig_cm)

    st.subheader("☁️ WordCloud từ đánh giá")
    all_text = " ".join(company_data['text_processed'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)

elif menu == "Phân cụm đánh giá":
    st.title("📌 Phân cụm đánh giá từ Review")
    st.markdown("Sử dụng TF-IDF và KMeans để nhóm các đánh giá tương đồng.")

    selected_company_cluster = st.selectbox("Chọn công ty để phân cụm:", company_list, key="cluster_company")
    company_cluster_data = Reviews[Reviews['Company Name'] == selected_company_cluster].copy()

    num_clusters = st.slider("Chọn số cụm (K):", 2, 10, 3)

    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(company_cluster_data['text_processed'])

    model_kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = model_kmeans.fit_predict(X_tfidf)
    company_cluster_data['Cluster'] = clusters

    st.subheader("Kết quả phân cụm")
    st.dataframe(company_cluster_data[['combined_text', 'Cluster']])

    st.subheader("Biểu đồ PCA 2D cụm")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_tfidf.toarray())
    company_cluster_data['PCA1'] = X_pca[:, 0]
    company_cluster_data['PCA2'] = X_pca[:, 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=company_cluster_data, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', ax=ax)
    st.pyplot(fig)

    st.info("Dựa trên các cụm, công ty có thể hiểu đánh giá chung và cải thiện chất lượng làm việc.")
