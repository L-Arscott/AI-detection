import pandas as pd
import string
from math import log
from sklearn.model_selection import train_test_split


def create_class_df(q_and_a_df):
    # Takes dataframe of questions and answers (both human and AI generated)
    # Returns a new dataframe with answers and their classification
    human_df = pd.DataFrame(q_and_a_df['Accepted answer'])
    human_df = human_df.rename(columns={'Accepted answer': 'Answer'})
    human_df['Class'] = 0

    ai_df = pd.DataFrame(q_and_a_df['GPT answer'][q_and_a_df['GPT answer'].notnull()])
    ai_df = ai_df.rename(columns={'GPT answer': 'Answer'})
    ai_df['Class'] = 1

    class_df = pd.concat([human_df, ai_df], ignore_index=True)

    return class_df


class NaiveBayes:
    """Naive Bayes model

    Parameters
    ----------
    class_df: pandas.core.frame.DataFrame
        Dataframe containing "Answers" column and "Class" column.


    Attributes
    ----------
    class_df: pandas.core.frame.DataFrame
        Dataframe containing "Answers" column and "Class" column.
    human_counts: dict
        Dictionary containing all words in text classed as human as keys. Values give the word count.
    ai_counts: dict
        As above for AI-classified text.
    n_words_human: int
        Total number of words in human-generated text.
    n_words_ai: int
        As above for AI.
    full_dict: set
        Contains all unique words from all text.
    n_words: int
        Number of words in full_dict.

    Methods
    -------
    classify
    """

    def __init__(self, class_df):
        self.class_df = class_df
        self.human_counts, self.ai_counts, self.n_words_human, self.n_words_ai, self.full_dict, self.n_words =\
            self._obtain_word_counts()

    def classify(self, text: str):
        # Given some text, classifies as either AI or human generated
        formatted_text = self._format_string(text)
        word_list = formatted_text.split()

        p_human = self._calculate_p_human_word_bag(word_list)
        p_ai = self._calculate_p_ai_word_bag(word_list)

        return int(p_human < p_ai)

    @staticmethod
    def _format_string(s):
        s = s.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        s = s.lower()  # Make all lower case

        return s

    def _obtain_word_counts(self):
        # Used to initiate class attributes
        human_df = self.class_df[self.class_df['Class'] == 0]
        ai_df = self.class_df[self.class_df['Class'] == 1]

        human_big_text = ' '.join(human_df['Answer'])
        ai_big_text = ' '.join(ai_df['Answer'])

        n_words_human = len(human_big_text)
        n_words_ai = len(ai_big_text)

        human_counts = {}
        for word in self._format_string(human_big_text).split():
            human_counts[word] = human_counts.get(word, 0) + 1

        ai_counts = {}
        for word in self._format_string(ai_big_text).split():
            ai_counts[word] = ai_counts.get(word, 0) + 1

        full_dict = set(list(human_counts.keys()) + list(ai_counts.keys()))
        n_words = len(full_dict)

        return human_counts, ai_counts, n_words_human, n_words_ai, full_dict, n_words

    def _calculate_p_human_word_bag(self, word_list):
        # Given a list of words, calculates their weight (human generated)
        p_list = [self._calculate_log_p_human_word(word) for word in word_list]

        return sum(p_list)

    def _calculate_p_ai_word_bag(self, word_list):
        # Given a list of words, calculates their weight (AI generated)
        p_list = [self._calculate_log_p_ai_word(word) for word in word_list]

        return sum(p_list)

    def _calculate_log_p_human_word(self, word: str, alpha=1):
        # Given a word, calculates its log-weight (human generated)
        n_occurrence = self.human_counts[word] if word in self.human_counts else 0

        p_human = (n_occurrence + alpha) / (self.n_words_human + alpha * self.n_words)

        return log(p_human)

    def _calculate_log_p_ai_word(self, word: str, alpha=1):
        # Given a word, calculates its log-weight (AI generated)
        n_occurrence = self.ai_counts[word] if word in self.ai_counts else 0

        p_ai = (n_occurrence + alpha) / (self.n_words_ai + alpha * self.n_words)

        return log(p_ai)


## Gather data
our_data_csv = 'SE_with_GPT_0_to_234.csv'
our_df = pd.read_csv(our_data_csv)
our_class_df = create_class_df(our_df)
our_train_df, our_test_df = train_test_split(our_class_df, test_size=0.3)

## Train model
nb = NaiveBayes(our_train_df)

## Calculate predictions and print accuracy
our_test_df["Prediction"] = our_class_df.apply(lambda row: nb.classify(row['Answer']), axis=1)
print(f'Accuracy is {sum(our_test_df["Class"] == our_test_df["Prediction"]) / len(our_test_df)}')

## Analysis: what words to humans prefer ? What words does AI prefer ?
table_data = []

for key in nb.full_dict:
    table_data.append({
        'word': key,
        'human count': nb.human_counts.get(key, 0),
        'AI count': nb.ai_counts.get(key, 0),
        'Ratio': nb.human_counts.get(key, 1) / nb.ai_counts.get(key, 1)
    })

# Create a Pandas DataFrame
df = pd.DataFrame(table_data)
df = df.sort_values(by='Ratio', ascending=False)

# Display the DataFrame
print(df)
