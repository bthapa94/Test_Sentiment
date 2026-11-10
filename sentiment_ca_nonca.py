import argparse
import string
from pathlib import Path

import nltk
import pandas as pd
import torch
from flair.data import Sentence
from flair.models import TextClassifier
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / 'data' / 'company_class_split.xlsx'
LM_FILE = BASE / 'data' / 'Loughran-McDonald_MasterDictionary_1993-2025.csv'
OUT_DIR = BASE / 'output'

FINBERT_MODEL = 'yiyanghkust/finbert-tone'
BATCH_SIZE = 32

for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'vader_lexicon']:
    nltk.download(pkg, quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def load_data(path, n=None):
    finalnonca = pd.read_excel(path, sheet_name=0)
    finalca = pd.read_excel(path, sheet_name=1)
    if n:
        finalca = finalca[finalca['companyid'].isin(finalca['companyid'].unique()[:n])]
        finalnonca = finalnonca[finalnonca['companyid'].isin(finalnonca['companyid'].unique()[:n])]
    print(f"finalca: {finalca['companyid'].nunique()} companies, {len(finalca)} rows")
    print(f"finalnonca: {finalnonca['companyid'].nunique()} companies, {len(finalnonca)} rows")
    return finalca, finalnonca


def clean_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    return ' '.join(lemmatizer.lemmatize(t) for t in tokens)


def company_texts(df):
    for cid, grp in df.groupby('companyid', sort=False):
        text = ' '.join(grp['componenttext'].dropna().astype(str))
        yield cid, grp['companyname'].iloc[0], text


def save(ca_df, nonca_df, out_dir, filename):
    path = out_dir / filename
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        ca_df.to_excel(writer, sheet_name='finalca', index=False)
        nonca_df.to_excel(writer, sheet_name='finalnonca', index=False)
    print('saved', path)


# VADER
def run_vader(df):
    sia = SentimentIntensityAnalyzer()
    rows = []
    for cid, name, text in company_texts(df):
        s = sia.polarity_scores(clean_text(text))
        rows.append({
            'companyid': cid,
            'companyname': name,
            'positive_score': s['pos'],
            'negative_score': s['neg'],
            'neutral_score': s['neu'],
            'compound_score': s['compound'],
            'label': 1 if s['compound'] < 0 else 0,
        })
    return pd.DataFrame(rows)


# TextBlob
def run_textblob(df):
    rows = []
    for cid, name, text in company_texts(df):
        blob = TextBlob(text)
        rows.append({
            'companyid': cid,
            'companyname': name,
            'polarity': blob.polarity,
            'subjectivity': blob.subjectivity,
            'label': 1 if blob.polarity < 0 else 0,
            'word_count': len(text.split()),
        })
    return pd.DataFrame(rows)


# Flair
def run_flair(df, clf):
    rows = []
    for cid, name, text in company_texts(df):
        cleaned = clean_text(text)
        sent = Sentence(cleaned)
        clf.predict(sent)
        tag = sent.labels[0].value
        conf = sent.labels[0].score
        rows.append({
            'companyid': cid,
            'companyname': name,
            'positive_score': conf if tag == 'POSITIVE' else 0,
            'negative_score': conf if tag == 'NEGATIVE' else 0,
            'neutral_score': 0,
            'compound_score': 0,
            'label': 1 if tag == 'NEGATIVE' else 0,
            'word_count': len(cleaned.split()),
        })
    return pd.DataFrame(rows)


# FinBERT (Huang, Wang & Yang)
class FinBert:
    def __init__(self, model_name=FINBERT_MODEL):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.idx = {lab.lower(): i for i, lab in self.model.config.id2label.items()}
        print('finbert running on', self.device)

    def probs(self, sentences):
        order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
        out = torch.zeros(len(sentences), len(self.idx))
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]
            inputs = self.tokenizer([sentences[i] for i in ids], return_tensors='pt',
                                    padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                out[ids] = torch.softmax(self.model(**inputs).logits, dim=1).cpu()
        return out


def run_finbert(df, fb):
    rows = []
    for cid, name, text in company_texts(df):
        sentences = [s for s in nltk.sent_tokenize(text) if s.strip()]
        probs = fb.probs(sentences)
        pred = probs.argmax(dim=1)

        n_pos = int((pred == fb.idx['positive']).sum())
        n_neg = int((pred == fb.idx['negative']).sum())
        n_neu = int((pred == fb.idx['neutral']).sum())
        tone = (n_pos - n_neg) / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0

        rows.append({
            'companyid': cid,
            'companyname': name,
            'positive_score': probs[:, fb.idx['positive']].mean().item(),
            'negative_score': probs[:, fb.idx['negative']].mean().item(),
            'neutral_score': probs[:, fb.idx['neutral']].mean().item(),
            'compound_score': tone,
            'n_sentences': len(sentences),
            'n_positive': n_pos,
            'n_negative': n_neg,
            'n_neutral': n_neu,
            'label': 1 if tone < 0 else 0,
            'word_count': len(text.split()),
        })
        print(f'finbert {cid}: {len(sentences)} sentences, tone {tone:.3f}')
    return pd.DataFrame(rows)


# Loughran-McDonald
def load_lm(path):
    lm = pd.read_csv(path)
    pos = set(lm.loc[lm['Positive'] > 0, 'Word'].str.lower())
    neg = set(lm.loc[lm['Negative'] > 0, 'Word'].str.lower())
    return pos, neg


def run_lm(df, group, lm_pos, lm_neg):
    rows = []
    for cid, name, text in company_texts(df):
        tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
        pos = sum(t in lm_pos for t in tokens)
        neg = sum(t in lm_neg for t in tokens)
        score = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0
        rows.append({
            'companyid': cid,
            'companyname': name,
            'positivewordcount': pos,
            'negativewordcount': neg,
            'totalwords': len(tokens),
            'sentimentscore': score,
            'class': group,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=DATA_FILE, type=Path)
    parser.add_argument('--lm', default=LM_FILE, type=Path)
    parser.add_argument('--out', default=OUT_DIR, type=Path)
    parser.add_argument('--n', type=int, default=None, help='only run the first N companies per group')
    parser.add_argument('--models', nargs='+', default=['vader', 'textblob', 'flair', 'finbert', 'lm'])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    finalca, finalnonca = load_data(args.data, args.n)

    if 'vader' in args.models:
        save(run_vader(finalca), run_vader(finalnonca), args.out,
             'final_sentiment_scores_VADER_preprocessed.xlsx')

    if 'textblob' in args.models:
        save(run_textblob(finalca), run_textblob(finalnonca), args.out,
             'final_sentiment_TextBlob_results.xlsx')

    if 'flair' in args.models:
        clf = TextClassifier.load('en-sentiment')
        save(run_flair(finalca, clf), run_flair(finalnonca, clf), args.out,
             'final_sentiment_Flair_results.xlsx')

    if 'lm' in args.models:
        lm_pos, lm_neg = load_lm(args.lm)
        save(run_lm(finalca, 'CA', lm_pos, lm_neg), run_lm(finalnonca, 'NonCA', lm_pos, lm_neg),
             args.out, 'final_sentiment_LM_combined.xlsx')

    if 'finbert' in args.models:
        fb = FinBert()
        save(run_finbert(finalca, fb), run_finbert(finalnonca, fb), args.out,
             'final_sentiment_FinBERT_results.xlsx')


if __name__ == '__main__':
    main()
