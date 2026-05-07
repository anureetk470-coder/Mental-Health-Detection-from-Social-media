# Mental Health Detection from Social Media

This Python ML project predicts whether a social media post suggests one of four states:

- Stress
- Anxiety
- Depression
- Normal

The included `data/sample_mental_health_posts.csv` is a small starter dataset so the project runs immediately. For a larger dataset, use a public dataset such as Kaggle's "Mental Health Social Posts Dataset" by Shamim Hasan, which contains 1,000 anonymized synthetic posts labeled for mental health signals under an MIT license:

https://www.kaggle.com/datasets/shamimhasan8/mental-health-social-posts-dataset

## Project Flow

1. Load dataset
2. Clean and preprocess text
3. Extract NLP features
4. Train multiple ML models
5. Evaluate accuracy, classification report, and confusion matrix
6. Show visualizations
7. Use a Streamlit frontend for prediction and analysis

## NLP Preprocessing

The pipeline performs:

- Lowercasing
- URL removal
- Emoji and punctuation removal
- Tokenization
- Stopword removal
- Lemmatization using NLTK WordNet when available

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate models:

```bash
python src/train_model.py
```

Run the frontend:

```bash
streamlit run app.py
```

## Dataset Format

Your CSV should contain:

- `post`: social media text
- `label`: one of `stress`, `anxiety`, `depression`, `normal`

Example:

```csv
post,label
"I feel overwhelmed by deadlines.",stress
"Had a peaceful day with friends.",normal
```

## Important Note

This project is for education and research only. It is not a medical diagnosis tool. Any mental health concern should be handled by a qualified professional.
