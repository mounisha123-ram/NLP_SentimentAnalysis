import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ------------------ Setup ------------------
nltk_path = 'nltk_data'
if not os.path.exists(os.path.join(nltk_path, 'tokenizers', 'punkt')):
    nltk.download('punkt', download_dir=nltk_path)
    nltk.download('stopwords', download_dir=nltk_path)
nltk.data.path.append(nltk_path)

st.set_page_config(page_title="ChatGPT Review Dashboard", layout="wide")
st.title("💬🔍 ChatGPT Reviews Insights & Sentiment Analysis")

# ------------------ Load Cleaned Dataset ------------------
@st.cache_data
def load_clean_data():
    df = pd.read_csv("chatgpt_reviews - chatgpt_reviews.csv")
    
    # Make sure columns are lowercase and clean
    df.columns = df.columns.str.strip().str.lower()
    
    # Convert date if available
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Add review_length
    if 'review' in df.columns:
        df['review_length'] = df['review'].apply(lambda x: len(str(x)))

    # Clean review fallback
    if 'clean_review' not in df.columns and 'review' in df.columns:
        df['clean_review'] = df['review']

    # ✅ Add sentiment based on rating
    if 'sentiment' not in df.columns and 'rating' in df.columns:
        def assign_sentiment(rating):
            if rating in [1, 2]:
                return 0  # Negative
            elif rating == 3:
                return 1  # Neutral
            else:
                return 2  # Positive
        df['sentiment'] = df['rating'].apply(assign_sentiment)

    return df


df = load_clean_data()

# ------------------ Load ML Assets ------------------
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('svm_model.pkl', 'rb'))
y_test = pickle.load(open('y_test.pkl', 'rb'))
y_pred_svm = pickle.load(open('y_pred_svm.pkl', 'rb'))

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ------------------ Sidebar Navigation ------------------
st.sidebar.title("Select a Page")
page = st.sidebar.radio("Go to", ["EDA", "Sentiment Analysis", "Model Evaluation"])

# ------------------ Page 1: EDA ------------------
if page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    questions = {
        "1. What is the distribution of review ratings?": "q1",
        "2. Helpful reviews count": "q2",
        "3. Most common keywords": "q3",
        "4. Rating trend over time": "q4",
        "5. Ratings by location": "q5",
        "6. Platform ratings": "q6",
        "7. Verified vs Non-verified": "q7",
        "8. Review length per rating": "q8",
        "9. Common words in 1-star": "q9",
        "10. Ratings by version": "q10",
    }

    selected = st.selectbox("Select a question to Analysis:", list(questions.keys()))
    st.markdown("---")

    if questions[selected] == "q1":
        fig, ax = plt.subplots()
        total = len(df)
        ax = sns.countplot(data=df, x='rating', order=sorted(df['rating'].dropna().unique()), palette='viridis')
        for p in ax.patches:
            height = p.get_height()
            percent = (height / total) * 100
            ax.annotate(f'{percent:.1f}%', (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom')
        st.pyplot(fig)

    elif questions[selected] == "q2":
        if 'helpful_votes' in df.columns:
            threshold = 10
            counts = [(df['helpful_votes'] > threshold).sum(), (df['helpful_votes'] <= threshold).sum()]
            labels = ['👍 Helpful (>10)', '👎 Not Helpful (≤10)']
            fig, ax = plt.subplots()
            ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
            st.pyplot(fig)
        else:
            st.warning("❗ 'helpful_votes' column not found.")

    elif questions[selected] == "q3":
        sentiment_map = {1: 'Positive', 2: 'Neutral', 0: 'Negative'}
        reverse_map = {v: k for k, v in sentiment_map.items()}
        
        chosen_sentiment = st.selectbox("Choose Sentiment:", list(reverse_map.keys()))
        val = reverse_map[chosen_sentiment]

        if 'sentiment' in df.columns:
            sentiment_df = df[df['sentiment'] == val]

            if not sentiment_df.empty:
                text = sentiment_df['clean_review'].astype(str).str.cat(sep=" ")

                # --- WordCloud ---
                st.markdown(f"### ☁️ WordCloud for {chosen_sentiment} Sentiment")

                wc = WordCloud(width=800, height=400, min_font_size=10, background_color='white').generate(text)
                fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)

                # --- Top 30 Word Count Bar Chart ---
                st.markdown(f"### 📊 Top 30 Words in {chosen_sentiment} Sentiment")

                word_list = []
                for msg in sentiment_df['clean_review'].dropna().tolist():
                    word_list.extend(msg.split())

                common_words = pd.DataFrame(Counter(word_list).most_common(30), columns=['word', 'count'])

                fig_bar, ax_bar = plt.subplots(figsize=(12, 5))
                sns.barplot(x='word', y='count', data=common_words, ax=ax_bar, palette='Blues_d')
                plt.xticks(rotation=45)
                st.pyplot(fig_bar)

            else:
                st.warning(f"No data found for {chosen_sentiment} sentiment.")
        else:
            st.warning("❗ 'sentiment' column not found.")

    elif questions[selected] == "q4":
        if 'date' in df.columns:
            trend = df.groupby(df['date'].dt.to_period('M'))['rating'].mean().reset_index()
            trend['date'] = trend['date'].dt.to_timestamp()
            fig = px.line(trend, x='date', y='rating', markers=True)
            st.plotly_chart(fig)
        else:
            st.warning("❗ 'date' column not found.")

    elif questions[selected] == "q5":
        if 'location' in df.columns and 'rating' in df.columns:
            st.subheader("📍 Average Rating by Location (Choropleth Map)")

            # Choropleth Map
            loc_rating = df.groupby('location')['rating'].mean().reset_index()
            fig_map = px.choropleth(
                loc_rating,
                locations='location',
                locationmode='country names',
                color='rating',
                color_continuous_scale='Blues',
                title='🌍 Average Rating by Country'
            )
            fig_map.update_geos(showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")
            st.plotly_chart(fig_map)

            st.subheader("📊 Distribution of Ratings by Location (Stacked Bar Chart)")

            # Stacked Bar Chart
            rating_dist = df.groupby(['location', 'rating']).size().reset_index(name='count')
            fig_bar = px.bar(
                rating_dist,
                x='location',
                y='count',
                color='rating',
                title='📊 Distribution of Ratings by Location',
                labels={'count': 'Number of Ratings'},
                barmode='stack',
                color_continuous_scale='Viridis'
            )
            fig_bar.update_layout(xaxis_title='Location', yaxis_title='Count of Ratings')
            st.plotly_chart(fig_bar)

        else:
            st.warning("❗ Either 'location' or 'rating' column not found.")

    elif questions[selected] == "q6":
            if 'platform' in df.columns and 'rating' in df.columns:
                platform_avg = df.groupby('platform')['rating'].mean().reset_index()
    
                # Plot
                fig = px.bar(
                    platform_avg,
                    x='platform',
                    y='rating',
                    color='platform',
                    text='rating',
                    title='💻📱 Average Rating by Platform (Web vs Mobile)',
                    labels={'rating': 'Average Rating'},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
    
                # Styling
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(
                    showlegend=False,
                    yaxis_range=[0, 5],  # to keep scale consistent
                    plot_bgcolor='white',
                    xaxis_title='Platform',
                    yaxis_title='Average Rating',
                    bargap=0.3
                )
    
                st.plotly_chart(fig)
            else:
                st.warning("❗ 'platform' or 'rating' column not found in the dataset.")


    elif questions[selected] == "q7":
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        if 'verified_purchase' in df.columns:
            avg = df.groupby('verified_purchase')['rating'].mean().reset_index()

            # Safely get values
            yes = avg[avg['verified_purchase'] == 'Yes']['rating'].values[0]
            no = avg[avg['verified_purchase'] == 'No']['rating'].values[0]

            # Create two subplots
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "indicator"}, {"type": "indicator"}]]
            )

            # ✅ Verified Indicator
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=yes,
                delta={'reference': no},
                title={'text': "✅ Verified", 'font': {'size': 18}},
                gauge={'axis': {'range': [0, 5]}, 'bar': {'color': "green"}},
                domain={'x': [0, 1], 'y': [0, 1]}
            ), row=1, col=1)

            # ❌ Non-Verified Indicator
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=no,
                delta={'reference': yes},
                title={'text': "❌ Non-Verified", 'font': {'size': 18}},
                gauge={'axis': {'range': [0, 5]}, 'bar': {'color': "red"}},
                domain={'x': [0, 1], 'y': [0, 1]}
            ), row=1, col=2)

            fig.update_layout(height=400, margin=dict(t=50, b=10))

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("❗ 'verified_purchase' column not found.")

    elif questions[selected] == "q8":
        fig = px.box(df, x='rating', y='review_length', color='rating')
        st.plotly_chart(fig)

    elif questions[selected] == "q9":
        if 'rating' in df.columns and 'review' in df.columns:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            from collections import Counter
            from nltk.corpus import stopwords
            import nltk

            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))

            # Filter 1-star reviews and drop missing ones
            one_star = df[df['rating'] == 1]['review'].dropna()

            # WordCloud
            text = " ".join(one_star.tolist())
            if text.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(text)
                fig_wc, ax_wc = plt.subplots(figsize=(12, 5))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                ax_wc.set_title("☁️ WordCloud of Most Common Words in 1-Star Reviews")
                st.pyplot(fig_wc)

            # Bar chart for most common words
            all_words = [word for review in one_star for word in str(review).lower().split() if word not in stop_words]
            common = Counter(all_words).most_common(20)

            if common:
                words, counts = zip(*common)
                fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
                ax_bar.barh(words[::-1], counts[::-1], color='crimson')
                ax_bar.set_title("📊 Top 20 Most Frequent Words in 1-Star Reviews")
                ax_bar.set_xlabel("Frequency")
                st.pyplot(fig_bar)
            else:
                st.warning("😕 No valid words found in 1-star reviews.")
        else:
            st.warning("❗ Required columns 'rating' or 'review' not found.")


    elif questions[selected] == "q10":
        if 'version' in df.columns and 'rating' in df.columns:
            st.subheader("📱 Version-wise Average Rating and Review Count")

            # Group by version: get average rating and review count
            version_stats = df.groupby('version').agg(
                avg_rating=('rating', 'mean'),
                review_count=('rating', 'count')
            ).reset_index()

            # Create bar chart with color based on review count
            fig = px.bar(
                version_stats,
                x='version',
                y='avg_rating',
                color='review_count',
                text='avg_rating',
                title='📊 Average Rating by App Version with Number of Reviews',
                labels={'avg_rating': 'Average Rating', 'review_count': 'Number of Reviews'}
            )

            # Customize appearance
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(
                xaxis_title='Version',
                yaxis_title='Average Rating',
                coloraxis_colorbar=dict(title='Number of Reviews'),
                plot_bgcolor='white',
                bargap=0.3
            )

            st.plotly_chart(fig)

        else:
            st.warning("❗ 'version' or 'rating' column not found.")

    # ------------------ Page 2: Sentiment Prediction ------------------
elif page == "Sentiment Analysis":
        st.title("💬 Sentiment Analysis")

        ps = PorterStemmer()
        def transform_text(text):
            text = text.lower()
            text = nltk.word_tokenize(text)
            y = [i for i in text if i.isalnum()]
            y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
            y = [ps.stem(i) for i in y]
            return " ".join(y)

        review = st.text_area("Enter a review about ChatGPT:", height=200)
        if st.button("Predict Sentiment"):
            if review.strip() == "":
                st.warning("Please enter some text.")
            else:
                with st.spinner("Predicting..."):
                    transformed_review = transform_text(review)
                    vector_input = vectorizer.transform([transformed_review]).toarray()
                    prediction = model.predict(vector_input)[0]
                st.success(f"Predicted Sentiment: **{label_map[prediction]}**")

    # ------------------ Page 3: Model Evaluation ------------------
elif page == "Model Evaluation":
        st.title("📈 Model Evaluation")
        st.subheader("Model Used --> Support Vector Machine (SVM)")

        accuracy = accuracy_score(y_test, y_pred_svm)
        st.metric("🔢 Accuracy", f"{accuracy*100:.2f}%")

        report = classification_report(y_test, y_pred_svm, target_names=['Negative', 'Neutral', 'Positive'])
        st.text("Classification Report:")
        st.text(report)

        cm = confusion_matrix(y_test, y_pred_svm)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'], ax=ax)
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
