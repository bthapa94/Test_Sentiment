arnings Call Sentiment: Class Action vs. Non-Class Action Firms

This repo scores the tone of earnings call transcripts for two groups of companies:

CA: firms that were hit with a securities class action lawsuit
Non-CA: a matched sample of firms that were not

For every company, all of its transcript text is combined and scored with five sentiment methods. The goal is to compare tone between the two groups.

Method	Type	Text used
VADER	Lexicon (NLTK)	Cleaned
TextBlob	Lexicon	Raw
Flair	Pre-trained model (en-sentiment)	Cleaned
Loughran-McDonald	Finance word lists	Raw words
FinBERT	Finance language model (yiyanghkust/finbert-tone)	Raw, sentence by sentence

"Cleaned" means lowercased, with punctuation and stopwords removed, and lemmatized.

Repo structure
stic/
├── sentiment_ca_nonca.py       # main script
├── sentiment_ca_nonca.ipynb    # same code as a notebook (Jupyter / Colab)
├── requirements.txt
├── README.md
├── data/                       # you add this, not tracked by git
│   ├── company_class_split.xlsx
│   └── Loughran-McDonald_MasterDictionary_1993-2025.csv
└── output/                     # created when you run the script
Getting started
1. Clone the repo
bash
git clone https://github.com/<your-username>/stic.git
cd stic
2. Install the packages

Python 3.10 to 3.12 is recommended.

bash
pip install -r requirements.txt

The first run also downloads the NLTK data, the Flair model and the FinBERT model. Allow a few minutes for that.

3. Add the data

Create a data/ folder and put two files in it:

company_class_split.xlsx

Sheet 1: non-CA companies
Sheet 2: CA companies
Required columns: companyid, companyname, componenttext

Loughran-McDonald_MasterDictionary_1993-2025.csv Download it from the Software Repository for Accounting and Finance.

The data is not part of this repo.

4. Run a quick test

Start with 5 companies from each group to make sure everything works:

bash
python sentiment_ca_nonca.py --n 5 --out output_test
5. Run the full dataset
bash
python sentiment_ca_nonca.py

Results are saved to output/.

Options
Option	What it does	Example
--n	Only use the first N companies in each group	--n 5
--models	Run only some of the models	--models vader lm
--out	Output folder	--out results
--data	Path to the transcript Excel file	--data D:/files/split.xlsx
--lm	Path to the LM dictionary	--lm D:/files/LM.csv

Model names for --models: vader, textblob, flair, lm, finbert.

Output

Each method writes one Excel file with two sheets, finalca and finalnonca. Every row is one company.

File	Main columns
final_sentiment_scores_VADER_preprocessed.xlsx	positive_score, negative_score, neutral_score, compound_score, label
final_sentiment_TextBlob_results.xlsx	polarity, subjectivity, label, word_count
final_sentiment_Flair_results.xlsx	positive_score, negative_score, label, word_count
final_sentiment_LM_combined.xlsx	positivewordcount, negativewordcount, totalwords, sentimentscore, class
final_sentiment_FinBERT_results.xlsx	positive_score, negative_score, neutral_score, compound_score, n_sentences, n_positive, n_negative, n_neutral, label

label: 1 = negative tone, 0 = not negative.

How each score is calculated

VADER: the compound score from NLTK's VADER, from −1 to +1. The label is 1 if compound < 0.

TextBlob: polarity from −1 to +1 and subjectivity from 0 to 1. The label is 1 if polarity < 0.

Flair: predicts POSITIVE or NEGATIVE with a confidence score. Flair has no neutral class.

Loughran-McDonald: counts words from the LM positive and negative lists.

sentimentscore = (positive words − negative words) / (positive words + negative words)

A word is in a list if its Positive / Negative value in the master dictionary is greater than 0. Those columns hold the year a word was added, and negative values mark words that were later removed.

FinBERT: uses the model from Huang, Wang & Yang (2023), which was trained on single sentences. Each sentence is classified as positive, negative or neutral, then rolled up to the company level:

compound_score = (positive sentences − negative sentences) / (positive + negative sentences)

positive_score, negative_score and neutral_score are the average probabilities across all of a company's sentences.

Running time

VADER, TextBlob, Flair and LM finish quickly. FinBERT scores every sentence, so it is slow on a CPU, and the full dataset can take many hours. If you have an NVIDIA GPU the script uses it automatically. On Google Colab, open the notebook and go to Runtime → Change runtime type → GPU.

To run FinBERT on its own:

bash
python sentiment_ca_nonca.py --models finbert
Troubleshooting

ModuleNotFoundError Run pip install -r requirements.txt again.

sentencepiece fails to build on Windows

bash
python -m pip install --upgrade pip
python -m pip install "sentencepiece>=0.2.0"
python -m pip install -r requirements.txt

cannot import name 'BucketNotFoundError'

bash
python -m pip install -U huggingface_hub datasets transformers flair

FileNotFoundError for the data Check that both files are inside data/ with the exact names shown above, or pass the paths with --data and --lm.

References
Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. Proceedings of ICWSM.
Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. Journal of Finance, 66(1), 35–65.
Akbik, A., Blythe, D., & Vollgraf, R. (2018). Contextual string embeddings for sequence labeling. Proceedings of COLING.
Huang, A. H., Wang, H., & Yang, Y. (2023). FinBERT: A large language model for extracting information from financial text. Contemporary Accounting Research, 40(2), 806–841.
