# TikTok App Review Sentiment Analysis

An NLP project that analyzes over 460,000 TikTok app store reviews to understand user sentiment, comparing a lexicon-based approach (VADER) against a supervised machine learning model (TF-IDF + Multinomial Naive Bayes) trained on actual star ratings.

## Business Problem

App store reviews are a goldmine of unfiltered user feedback, but at scale they're impossible to read manually. This project automates sentiment classification of TikTok reviews to surface how users actually feel about the app, and tests whether a model trained on real star ratings can predict sentiment more reliably than a generic, out-of-the-box sentiment tool.

## Dataset

460,287 TikTok app reviews, each with:
- `reviewId` — unique review identifier
- `content` — the review text
- `score` — the user's star rating (1–5)

After removing rows with missing review text, 460,256 reviews remained for analysis.

## Approach

1. **Data cleaning** — dropped rows with missing review content.
2. **Text preprocessing** — lowercased text, removed URLs and non-alphabetic characters, applied lemmatization (via NLTK's WordNetLemmatizer), and removed both standard English stopwords and custom domain-specific stopwords (e.g. "tiktok," "app," "video") that would otherwise dominate every review.
3. **Lexicon-based sentiment (VADER)** — applied NLTK's VADER sentiment analyzer to the cleaned text, classifying each review as Positive, Negative, or Neutral based on its compound sentiment score.
4. **Ground-truth labeling** — converted the 5-star rating scale into 3 sentiment classes to use as ground truth: 4–5 stars = Positive, 3 stars = Neutral, 1–2 stars = Negative.
5. **Supervised model** — vectorized cleaned review text with TF-IDF (top 5,000 features) and trained a Multinomial Naive Bayes classifier to predict sentiment from the rating-based labels, using an 80/20 stratified train/test split.
6. **Evaluation** — compared training vs. test accuracy to check for overfitting, and reviewed precision/recall/F1 per sentiment class.
7. **Visualization** — sentiment distribution pie charts, top-10 frequent word bar chart and word cloud, and stacked bar charts comparing sentiment breakdown by original star rating (for both VADER and the trained model).

## Results

**VADER sentiment distribution** (lexicon-based, unsupervised):

| Sentiment | Share |
|---|---|
| Positive | 64.93% |
| Neutral | 27.77% |
| Negative | 7.30% |

**Multinomial Naive Bayes model** (trained on star-rating-derived labels):

| Metric | Score |
|---|---|
| Test Accuracy | 84.39% |
| Training Accuracy | 84.66% |

The close gap between training and test accuracy suggests the model generalizes well without significant overfitting. Performance was strongest on the Positive class (F1 = 0.91) and weaker on Negative (F1 = 0.55) and especially Neutral (F1 = 0.004) — the model struggles to distinguish Neutral reviews, largely because Neutral is a small, ambiguous middle category between clearly positive and clearly negative feedback, and the dataset is heavily imbalanced toward Positive reviews.

## Tech Stack

- **Python**: pandas
- **NLP**: NLTK (VADER, stopwords, WordNetLemmatizer), TF-IDF (scikit-learn)
- **Modeling**: scikit-learn (Multinomial Naive Bayes)
- **Visualization**: matplotlib, seaborn, WordCloud
- **Environment**: Google Colab

## Repository Structure

```
├── Tiktok_Sentiment_Analysis.ipynb   # Full analysis notebook (preprocessing, VADER, ML model, visualizations)
└── README.md                          # Project documentation
```

*Note: the raw dataset (`TikTok.csv`) is not included in this repository due to its size (~460K rows). To run the notebook, place `TikTok.csv` — containing `reviewId`, `content`, and `score` columns — in the same directory as the notebook.*

## How to Run

1. Clone this repository
2. Open `Tiktok_Sentiment_Analysis.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab
3. Install dependencies: `pip install pandas nltk scikit-learn matplotlib seaborn wordcloud`
4. Add `TikTok.csv` to the working directory
5. Run all cells (the notebook auto-downloads required NLTK resources on first run)
