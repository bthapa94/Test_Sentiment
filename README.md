# Test_Sentiment
Sentiment of earnings calls: class action vs. non-class action firms

Company-level sentiment scores for firms that faced a securities class action (CA) and a matched sample of firms that did not (non-CA). All of a company's earnings call text (componenttext) is combined and scored with five methods:

Method	What it uses	Output file
VADER	NLTK lexicon, cleaned text	final_sentiment_scores_VADER_preprocessed.xlsx
TextBlob	pattern lexicon, raw text	final_sentiment_TextBlob_results.xlsx
Flair	en-sentiment model, cleaned text	final_sentiment_Flair_results.xlsx
Loughran-McDonald	LM master dictionary word counts	final_sentiment_LM_combined.xlsx
FinBERT	yiyanghkust/finbert-tone, sentence level	final_sentiment_FinBERT_results.xlsx

Each output file has two sheets, finalca and finalnonca, with one row per company.

Setup
pip install -r requirements.txt

Python 3.10–3.12. A GPU is optional but FinBERT is much faster with one.

Data

Data files are not included in the repo. Put them in a data/ folder:

data/
├── company_class_split.xlsx                              # sheet 1 = non-CA, sheet 2 = CA
└── Loughran-McDonald_MasterDictionary_1993-2025.csv

company_class_split.xlsx needs the columns companyid, companyname and componenttext.

The LM dictionary can be downloaded from the Software Repository for Accounting and Finance.

Running
python sentiment_ca_nonca.py

Quick test on the first 5 companies of each group:

python sentiment_ca_nonca.py --n 5 --out output_test

Only some of the models:

python sentiment_ca_nonca.py --models vader lm

Other paths can be passed with --data, --lm and --out.

sentiment_ca_nonca.ipynb has the same code split into cells, for Jupyter or Colab.

Notes on the measures
Label: 1 = negative, 0 = not negative. VADER uses compound < 0, TextBlob polarity < 0, Flair a NEGATIVE prediction, LM and FinBERT tone < 0.
Loughran-McDonald: score = (positive − negative) / (positive + negative), using words with a value > 0 in the Positive / Negative columns of the master dictionary (those columns hold the year a word was added; negative values are words that were later removed). Words are matched as they appear, without lemmatizing.
FinBERT: the model from Huang, Wang & Yang (2023) is trained on single sentences, so each sentence is scored separately. positive_score / negative_score / neutral_score are the average probabilities across a company's sentences, and compound_score is (positive sentences − negative sentences) / (positive + negative sentences).
References
Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. ICWSM.
Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. Journal of Finance, 66(1), 35–65.
Akbik, A., Blythe, D., & Vollgraf, R. (2018). Contextual string embeddings for sequence labeling. COLING.
Huang, A. H., Wang, H., & Yang, Y. (2023). FinBERT: A large language model for extracting information from financial text. Contemporary Accounting Research, 40(2), 806–841.
