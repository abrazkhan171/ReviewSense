import re
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "sentiment_model.joblib"
METRICS_PATH = BASE / "models" / "metrics.joblib"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ReviewSense",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        color: #666;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #dddddd;
        margin-top: 20px;
        background-color: #fafafa;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        text-align: center;
        background-color: #fafafa;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z\s']",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# RATING → SENTIMENT
# ============================================================

def rating_to_sentiment(rating):

    try:
        rating = float(rating)
    except Exception:
        return np.nan

    if rating <= 2:
        return "Negative"

    elif rating == 3:
        return "Neutral"

    elif rating >= 4:
        return "Positive"

    return np.nan


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD SAVED TRAINING METRICS
# ============================================================

@st.cache_resource
def load_metrics():

    if not METRICS_PATH.exists():
        return None

    return joblib.load(METRICS_PATH)


model = load_model()
saved_metrics = load_metrics()


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_df" not in st.session_state:
    st.session_state["analysis_df"] = None

if "performance_data" not in st.session_state:
    st.session_state["performance_data"] = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 ReviewSense")

st.sidebar.markdown(
    "### Sentiment Analysis System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Single Review Analysis",
        "Bulk Analysis",
        "Visual Analytics",
        "Model Performance",
    ],
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">ReviewSense</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Sentiment Analysis on Reviews'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHECK MODEL
# ============================================================

if model is None:

    st.error(
        "Sentiment model not found. "
        "Please run `python train_model.py` first."
    )

    st.stop()


# ============================================================
# 1. SINGLE REVIEW ANALYSIS
# ============================================================

if page == "Single Review Analysis":

    st.header("🔍 Single Review Analysis")

    st.write(
        "Enter a review below and the trained machine learning "
        "model will predict its sentiment."
    )

    review = st.text_area(
        "Enter your review",
        placeholder=(
            "Example: The food was amazing and "
            "the service was excellent."
        ),
        height=150,
    )

    analyze_button = st.button(
        "🔎 Analyze Review",
        type="primary",
    )

    if analyze_button:

        if not review.strip():

            st.warning(
                "Please enter a review before analyzing."
            )

        else:

            cleaned_review = clean_text(review)

            prediction = model.predict(
                [cleaned_review]
            )[0]

            probabilities = model.predict_proba(
                [cleaned_review]
            )[0]

            classes = model.classes_

            confidence = float(
                np.max(probabilities)
            )

            # ------------------------------------------------
            # EMOJI
            # ------------------------------------------------

            if prediction == "Positive":
                emoji = "😊"

            elif prediction == "Negative":
                emoji = "😞"

            else:
                emoji = "😐"

            # ------------------------------------------------
            # KEY CONTRIBUTING WORDS
            # ------------------------------------------------

            key_words = []

            try:
                tfidf = model.named_steps["tfidf"]
                classifier = model.named_steps["classifier"]

                vector = tfidf.transform([cleaned_review])
                class_index = list(classifier.classes_).index(prediction)
                contributions = (
                    vector.multiply(classifier.coef_[class_index])
                    .tocsr()
                )

                feature_names = tfidf.get_feature_names_out()

                ranked = sorted(
                    zip(contributions.indices, contributions.data),
                    key=lambda item: item[1],
                    reverse=True,
                )

                for index, score in ranked:
                    word = feature_names[index]
                    if (
                        score > 0
                        and " " not in word
                        and word not in ENGLISH_STOP_WORDS
                    ):
                        key_words.append(word)
                    if len(key_words) == 4:
                        break

            except Exception:
                key_words = []

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            colors = {
                "Positive": ("#16a34a", "#22c55e", "#ecfdf3", "#bbf7d0", "#d1fae5"),
                "Negative": ("#dc2626", "#ef4444", "#fef2f2", "#fecaca", "#fee2e2"),
            }

            text_color, accent, background, border, chip = colors.get(
                prediction,
                ("#b45309", "#f59e0b", "#fffbeb", "#fde68a", "#fef3c7"),
            )

            confidence_pct = confidence * 100

            chips_html = "".join(
                f'<span style="background:{chip};color:{text_color};'
                f'padding:6px 14px;border-radius:999px;font-size:14px;'
                f'margin:0 8px 8px 0;display:inline-block;">{word}</span>'
                for word in key_words
            )

            words_html = (
                '<div style="margin-top:18px;font-size:15px;color:#333;">'
                'Key contributing words:</div>'
                f'<div style="margin-top:10px;">{chips_html}</div>'
                if key_words else ""
            )

            result_html = (
                f'<div style="padding:22px;border-radius:14px;'
                f'border:1px solid {border};background:{background};margin-top:20px;">'
                '<div style="display:flex;align-items:center;gap:22px;">'
                f'<div style="width:64px;height:64px;border-radius:50%;'
                f'background:{accent};display:flex;align-items:center;'
                f'justify-content:center;font-size:36px;flex-shrink:0;">{emoji}</div>'
                '<div style="flex:1;">'
                f'<div style="font-size:26px;font-weight:700;color:{text_color};">'
                f'{prediction}</div>'
                '<div style="font-size:15px;color:#333;margin:4px 0 10px;">'
                f'Confidence: {confidence_pct:.1f}%</div>'
                '<div style="display:flex;align-items:center;gap:12px;">'
                '<div style="flex:1;height:10px;background:#e5e7eb;'
                'border-radius:999px;overflow:hidden;">'
                f'<div style="width:{confidence_pct:.1f}%;height:100%;'
                f'background:{accent};border-radius:999px;"></div></div>'
                f'<span style="font-size:14px;color:#333;">{confidence_pct:.1f}%</span>'
                '</div></div></div>'
                f'{words_html}'
                '</div>'
            )

            st.markdown(
                result_html,
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            st.subheader(
                "Sentiment Probabilities"
            )

            probability_df = pd.DataFrame(
                {
                    "Sentiment": classes,
                    "Probability": probabilities * 100,
                }
            )

            probability_df = probability_df.sort_values(
                "Probability",
                ascending=False,
            )

            st.dataframe(
                probability_df.style.format(
                    {
                        "Probability": "{:.2f}%"
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            # ------------------------------------------------
            # PROBABILITY CHART
            # ------------------------------------------------

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            ax.bar(
                probability_df["Sentiment"],
                probability_df["Probability"],
            )

            ax.set_ylabel(
                "Probability (%)"
            )

            ax.set_xlabel(
                "Sentiment"
            )

            ax.set_title(
                "Sentiment Prediction Probability"
            )

            ax.set_ylim(
                0,
                100
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


# ============================================================
# 2. BULK ANALYSIS
# ============================================================

elif page == "Bulk Analysis":

    st.header("📁 Bulk Review Analysis")

    st.write(
        "Upload a CSV file containing customer reviews. "
        "A rating column is optional but required for "
        "calculating actual model performance."
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help=(
            "Your CSV should contain a column named "
            "'review'. A rating column is optional."
        ),
    )

    # --------------------------------------------------------
    # SAVE UPLOADED FILE TO SESSION
    # --------------------------------------------------------

    if uploaded is not None:

        st.session_state["uploaded_csv_bytes"] = (
            uploaded.getvalue()
        )

        st.session_state["uploaded_csv_name"] = (
            uploaded.name
        )

    # --------------------------------------------------------
    # RESTORE FILE FROM SESSION
    # --------------------------------------------------------

    if "uploaded_csv_bytes" in st.session_state:

        try:

            df = pd.read_csv(
                io.BytesIO(
                    st.session_state[
                        "uploaded_csv_bytes"
                    ]
                )
            )

            filename = st.session_state.get(
                "uploaded_csv_name",
                "Uploaded CSV",
            )

            st.success(
                f"Dataset loaded: {filename}"
            )

            # ------------------------------------------------
            # FIND REVIEW COLUMN
            # ------------------------------------------------

            review_column = None

            possible_review_columns = [
                "review",
                "reviews",
                "text",
                "comment",
                "comments",
                "review_text",
                "reviewtext",
            ]

            for column in df.columns:

                if column.lower().strip() in possible_review_columns:

                    review_column = column
                    break

            # If no standard review column exists,
            # look for object/string column

            if review_column is None:

                text_columns = df.select_dtypes(
                    include=["object"]
                ).columns

                if len(text_columns) > 0:
                    review_column = text_columns[0]

            # ------------------------------------------------
            # FIND RATING COLUMN
            # ------------------------------------------------

            rating_column = None

            possible_rating_columns = [
                "rating",
                "ratings",
                "score",
                "stars",
                "review_rating",
                "reviewrating",
            ]

            for column in df.columns:

                if column.lower().strip() in possible_rating_columns:

                    rating_column = column
                    break

            # ------------------------------------------------
            # CHECK REVIEW COLUMN
            # ------------------------------------------------

            if review_column is None:

                st.error(
                    "No review/text column could be detected."
                )

                st.info(
                    "Please use a column such as "
                    "'review', 'reviews', 'text' or 'comment'."
                )

            else:

                st.info(
                    f"Review column detected: "
                    f"**{review_column}**"
                )

                if rating_column:

                    st.info(
                        f"Rating column detected: "
                        f"**{rating_column}**"
                    )

                # ------------------------------------------------
                # ANALYZE BUTTON
                # ------------------------------------------------

                analyze_bulk = st.button(
                    "🚀 Analyze Dataset",
                    type="primary",
                )

                # ------------------------------------------------
                # RUN ANALYSIS
                # ------------------------------------------------

                if (
                    analyze_bulk
                    or st.session_state["analysis_df"]
                    is None
                ):

                    with st.spinner(
                        "Analyzing reviews..."
                    ):

                        result_df = df.copy()

                        cleaned_reviews = (
                            result_df[
                                review_column
                            ]
                            .fillna("")
                            .astype(str)
                            .map(clean_text)
                        )

                        predictions = model.predict(
                            cleaned_reviews
                        )

                        probabilities = (
                            model.predict_proba(
                                cleaned_reviews
                            )
                        )

                        confidence = (
                            probabilities.max(
                                axis=1
                            )
                        )

                        result_df[
                            "Sentiment"
                        ] = predictions

                        result_df[
                            "Confidence"
                        ] = (
                            confidence * 100
                        ).round(2)

                        # ----------------------------------------
                        # ACTUAL SENTIMENT FROM RATINGS
                        # ----------------------------------------

                        if rating_column:

                            result_df[
                                "Actual Sentiment"
                            ] = result_df[
                                rating_column
                            ].apply(
                                rating_to_sentiment
                            )

                            valid_df = (
                                result_df[
                                    result_df[
                                        "Actual Sentiment"
                                    ].notna()
                                ]
                            )

                            if len(valid_df) > 0:

                                st.session_state[
                                    "performance_data"
                                ] = {

                                    "actual":
                                        valid_df[
                                            "Actual Sentiment"
                                        ].tolist(),

                                    "predicted":
                                        valid_df[
                                            "Sentiment"
                                        ].tolist(),

                                    "total":
                                        len(valid_df),

                                    "rating_column":
                                        rating_column,

                                    "review_column":
                                        review_column,
                                }

                        else:

                            st.session_state[
                                "performance_data"
                            ] = None

                        st.session_state[
                            "analysis_df"
                        ] = result_df

                # ------------------------------------------------
                # SHOW STORED RESULTS
                # ------------------------------------------------

                result_df = st.session_state[
                    "analysis_df"
                ]

                if result_df is not None:

                    st.subheader(
                        "📊 Analysis Summary"
                    )

                    total_reviews = len(
                        result_df
                    )

                    positive_count = int(
                        (
                            result_df[
                                "Sentiment"
                            ]
                            == "Positive"
                        ).sum()
                    )

                    neutral_count = int(
                        (
                            result_df[
                                "Sentiment"
                            ]
                            == "Neutral"
                        ).sum()
                    )

                    negative_count = int(
                        (
                            result_df[
                                "Sentiment"
                            ]
                            == "Negative"
                        ).sum()
                    )

                    col1, col2, col3, col4 = (
                        st.columns(4)
                    )

                    with col1:

                        st.metric(
                            "Total Reviews",
                            total_reviews,
                        )

                    with col2:

                        st.metric(
                            "😊 Positive",
                            positive_count,
                        )

                    with col3:

                        st.metric(
                            "😐 Neutral",
                            neutral_count,
                        )

                    with col4:

                        st.metric(
                            "😞 Negative",
                            negative_count,
                        )

                    # ------------------------------------------------
                    # PERCENTAGES
                    # ------------------------------------------------

                    st.subheader(
                        "Sentiment Distribution"
                    )

                    percentage_df = pd.DataFrame(
                        {
                            "Sentiment": [
                                "Positive",
                                "Neutral",
                                "Negative",
                            ],
                            "Count": [
                                positive_count,
                                neutral_count,
                                negative_count,
                            ],
                        }
                    )

                    percentage_df[
                        "Percentage"
                    ] = (
                        percentage_df["Count"]
                        / total_reviews
                        * 100
                    ).round(2)

                    st.dataframe(
                        percentage_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    # ------------------------------------------------
                    # ANALYZED DATA
                    # ------------------------------------------------

                    st.subheader(
                        "📋 Analyzed Reviews"
                    )

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        height=400,
                    )

                    # ------------------------------------------------
                    # DOWNLOAD
                    # ------------------------------------------------

                    csv_data = (
                        result_df.to_csv(
                            index=False
                        ).encode("utf-8")
                    )

                    st.download_button(
                        label="⬇️ Download Analyzed CSV",
                        data=csv_data,
                        file_name=(
                            "reviewsense_analyzed.csv"
                        ),
                        mime="text/csv",
                    )

                # ------------------------------------------------
                # REMOVE DATASET
                # ------------------------------------------------

                st.divider()

                if st.button(
                    "🗑️ Remove Uploaded CSV"
                ):

                    st.session_state.pop(
                        "uploaded_csv_bytes",
                        None,
                    )

                    st.session_state.pop(
                        "uploaded_csv_name",
                        None,
                    )

                    st.session_state.pop(
                        "analysis_df",
                        None,
                    )

                    st.session_state.pop(
                        "performance_data",
                        None,
                    )

                    st.rerun()

        except Exception as e:

            st.error(
                f"Error reading CSV: {e}"
            )


# ============================================================
# 3. VISUAL ANALYTICS
# ============================================================

elif page == "Visual Analytics":

    st.header("📈 Visual Analytics")

    result_df = st.session_state.get(
        "analysis_df"
    )

    if result_df is None:

        st.info(
            "Please upload and analyze a CSV "
            "from Bulk Analysis first."
        )

    else:

        # ----------------------------------------------------
        # SENTIMENT COUNTS
        # ----------------------------------------------------

        sentiment_counts = (
            result_df[
                "Sentiment"
            ]
            .value_counts()
            .reindex(
                [
                    "Positive",
                    "Neutral",
                    "Negative",
                ],
                fill_value=0,
            )
        )

        # ----------------------------------------------------
        # PIE CHART
        # ----------------------------------------------------

        st.subheader(
            "🥧 Sentiment Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(5, 3.5)
        )

        ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            startangle=90,
        )

        ax.set_title(
            "Overall Sentiment Distribution"
        )

        plt.tight_layout()

        st.pyplot(fig, width="content")

        plt.close(fig)

        # ----------------------------------------------------
        # BAR CHART
        # ----------------------------------------------------

        st.subheader(
            "📊 Sentiment Count"
        )

        fig, ax = plt.subplots(
            figsize=(6, 3)
        )

        ax.bar(
            sentiment_counts.index,
            sentiment_counts.values,
        )

        ax.set_xlabel(
            "Sentiment"
        )

        ax.set_ylabel(
            "Number of Reviews"
        )

        ax.set_title(
            "Number of Reviews by Sentiment"
        )

        plt.tight_layout()

        st.pyplot(fig, width="content")

        plt.close(fig)

        # ----------------------------------------------------
        # REVIEW LENGTH
        # ----------------------------------------------------

        st.subheader(
            "📝 Review Length Analysis"
        )

        review_column = None

        for column in result_df.columns:

            if column.lower().strip() in [
                "review",
                "reviews",
                "text",
                "comment",
                "comments",
                "review_text",
                "reviewtext",
            ]:

                review_column = column
                break

        if review_column:

            review_lengths = (
                result_df[
                    review_column
                ]
                .fillna("")
                .astype(str)
                .str.len()
            )

            fig, ax = plt.subplots(
                figsize=(6.5, 3)
            )

            ax.hist(
                review_lengths,
                bins=30,
            )

            ax.set_xlabel(
                "Review Length (characters)"
            )

            ax.set_ylabel(
                "Number of Reviews"
            )

            ax.set_title(
                "Distribution of Review Length"
            )

            plt.tight_layout()

            st.pyplot(fig, width="content")

            plt.close(fig)

        # ----------------------------------------------------
        # RATING VS SENTIMENT
        # ----------------------------------------------------

        rating_column = None

        for column in result_df.columns:

            if column.lower().strip() in [
                "rating",
                "ratings",
                "score",
                "stars",
                "review_rating",
                "reviewrating",
            ]:

                rating_column = column
                break

        if rating_column:

            st.subheader(
                "⭐ Rating vs Sentiment"
            )

            rating_sentiment = (
                result_df.groupby(
                    [
                        rating_column,
                        "Sentiment",
                    ]
                )
                .size()
                .reset_index(
                    name="Count"
                )
            )

            fig, ax = plt.subplots(
                figsize=(6.5, 3.5)
            )

            sns.barplot(
                data=rating_sentiment,
                x=rating_column,
                y="Count",
                hue="Sentiment",
                ax=ax,
            )

            ax.set_title(
                "Rating vs Predicted Sentiment"
            )

            ax.set_xlabel(
                "Rating"
            )

            ax.set_ylabel(
                "Number of Reviews"
            )

            plt.tight_layout()

            st.pyplot(fig, width="content")

            plt.close(fig)

        else:

            st.info(
                "No rating column found. "
                "Rating vs sentiment analysis is unavailable."
            )


# ============================================================
# 4. MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("🎯 Model Performance")

    performance_data = (
        st.session_state.get(
            "performance_data"
        )
    )

    # --------------------------------------------------------
    # DYNAMIC PERFORMANCE FROM UPLOADED DATASET
    # --------------------------------------------------------

    if performance_data is not None:

        actual = performance_data[
            "actual"
        ]

        predicted = performance_data[
            "predicted"
        ]

        total = performance_data[
            "total"
        ]

        rating_column = performance_data[
            "rating_column"
        ]

        review_column = performance_data[
            "review_column"
        ]

        st.success(
            f"Performance calculated from the uploaded dataset "
            f"using **{total:,} valid reviews**."
        )

        st.write(
            f"**Review column:** {review_column}"
        )

        st.write(
            f"**Rating column:** {rating_column}"
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            actual,
            predicted,
        )

        precision = precision_score(
            actual,
            predicted,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            actual,
            predicted,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            actual,
            predicted,
            average="weighted",
            zero_division=0,
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:

            st.metric(
                "Accuracy",
                f"{accuracy * 100:.2f}%",
            )

        with col2:

            st.metric(
                "Precision",
                f"{precision * 100:.2f}%",
            )

        with col3:

            st.metric(
                "Recall",
                f"{recall * 100:.2f}%",
            )

        with col4:

            st.metric(
                "F1 Score",
                f"{f1 * 100:.2f}%",
            )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        st.subheader(
            "Confusion Matrix"
        )

        labels = [
            "Positive",
            "Neutral",
            "Negative",
        ]

        cm = confusion_matrix(
            actual,
            predicted,
            labels=labels,
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )

        ax.set_xlabel(
            "Predicted Sentiment"
        )

        ax.set_ylabel(
            "Actual Sentiment"
        )

        ax.set_title(
            "Confusion Matrix"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # CLASSIFICATION REPORT
        # ----------------------------------------------------

        st.subheader(
            "Classification Report"
        )

        report = classification_report(
            actual,
            predicted,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.style.format(
                {
                    "precision": "{:.2f}",
                    "recall": "{:.2f}",
                    "f1-score": "{:.2f}",
                    "support": "{:.0f}",
                }
            ),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        st.info(
            "The uploaded dataset's ratings are converted into "
            "Actual Sentiment labels: 1–2 = Negative, "
            "3 = Neutral, and 4–5 = Positive. "
            "These actual labels are compared with the model's "
            "predictions to calculate Accuracy, Precision, "
            "Recall and F1 Score."
        )

    else:

        # ----------------------------------------------------
        # NO UPLOADED DATASET
        # ----------------------------------------------------

        st.warning(
            "No evaluated dataset is currently available."
        )

        st.write(
            "Upload a CSV containing both reviews and ratings "
            "in Bulk Analysis to calculate model performance "
            "on your own dataset."
        )

        # ----------------------------------------------------
        # SHOW ORIGINAL TRAINING METRICS AS REFERENCE
        # ----------------------------------------------------

        if saved_metrics is not None:

            st.divider()

            st.subheader(
                "📚 Original Training/Test Performance"
            )

            st.caption(
                "These are the metrics obtained when the model "
                "was trained and tested on the SST-5 dataset. "
                "They are shown only as reference."
            )

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:

                st.metric(
                    "Accuracy",
                    f"{saved_metrics['accuracy'] * 100:.2f}%",
                )

            with col2:

                st.metric(
                    "Precision",
                    f"{saved_metrics['precision'] * 100:.2f}%",
                )

            with col3:

                st.metric(
                    "Recall",
                    f"{saved_metrics['recall'] * 100:.2f}%",
                )

            with col4:

                st.metric(
                    "F1 Score",
                    f"{saved_metrics['f1'] * 100:.2f}%",
                )

            # ------------------------------------------------
            # ORIGINAL CONFUSION MATRIX
            # ------------------------------------------------

            if (
                "confusion_matrix"
                in saved_metrics
            ):

                st.subheader(
                    "Original Test Confusion Matrix"
                )

                labels = saved_metrics.get(
                    "labels",
                    [
                        "Positive",
                        "Neutral",
                        "Negative",
                    ],
                )

                cm = np.array(
                    saved_metrics[
                        "confusion_matrix"
                    ]
                )

                fig, ax = plt.subplots(
                    figsize=(7, 5)
                )

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=labels,
                    yticklabels=labels,
                    ax=ax,
                )

                ax.set_xlabel(
                    "Predicted Sentiment"
                )

                ax.set_ylabel(
                    "Actual Sentiment"
                )

                ax.set_title(
                    "Original Test Confusion Matrix"
                )

                plt.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

        else:

            st.info(
                "No saved model metrics were found."
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "ReviewSense • AI & Data Science Mini Project"
)

st.sidebar.caption(
    "TF-IDF + Logistic Regression"
)