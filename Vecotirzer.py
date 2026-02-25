import re # for tokenizing
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


positive_reviews = "rt-polarity.pos"
negative_reviews = "rt-polarity.neg"
SEED = 42 # for reproducability


def load_lines(path: str):
    with open(path, "r", encoding="latin-1") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines



# My attempt to create a manual vectorizer using Bag of words approach
class SimpleCountVectorizer:

    def __init__(self, min_df=2, max_df=None, max_features=None):
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.vocab_ = None  # token count at start

    def _tokenize(self, text: str):
        # simple tokenizer
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, X_texts): # initializing the counter
        df = Counter()
        for text in X_texts:
            toks = set(self._tokenize(text))  # count each token once
            df.update(toks)

        items = []
        for tok, d in df.items(): # each token and its document frequency
            if d < self.min_df:
                continue
            if self.max_df is not None and d > self.max_df:
                continue
            items.append((tok, d)) #add to counter if it passes the min_df and max_df filters

        # sort by document frequency descending, then token ascending
        items.sort(key=lambda x: (-x[1], x[0]))

        if self.max_features is not None:
            items = items[: self.max_features] #puts a cap on the number of features if max_features is set

        self.vocab_ = {tok: i for i, (tok, _) in enumerate(items)} # tells which token corresponds to which column in the output matrix
        return self

    def transform(self, X_texts):
        if self.vocab_ is None:
            raise ValueError("Vectorizer not fit yet. Call fit() first.")

        V = len(self.vocab_) #size of teh vocabulary
        X = np.zeros((len(X_texts), V), dtype=np.float32)
        # each row is a document and each column is a token in the vocab. The value is the count of that token in that document
        for i, text in enumerate(X_texts):
            counts = Counter(self._tokenize(text))
            for tok, c in counts.items():
                j = self.vocab_.get(tok)
                if j is not None:
                    X[i, j] = c
        return X

    def fit_transform(self, X_texts): #used for training
        return self.fit(X_texts).transform(X_texts)


# functions for printing split sizes and evaluation reports
def print_split_sizes(X_train, X_dev, X_test, y_train, y_dev, y_test):
    print("Split sizes:")
    print(f"  Train set: {len(X_train)}  (pos={int(y_train.sum())}, neg={int((y_train==0).sum())})")
    print(f"  Development set:   {len(X_dev)}   (pos={int(y_dev.sum())},   neg={int((y_dev==0).sum())})")
    print(f"  Test set:  {len(X_test)}  (pos={int(y_test.sum())},  neg={int((y_test==0).sum())})")

#prints title accuracy and classification report for the given true and predicted labels
def eval_and_print(y_true, y_pred, title):
    print(f"\n{title}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

def main():
    # Loading the dataset
    pos = load_lines(positive_reviews)
    neg = load_lines(negative_reviews)

    texts = np.array(pos + neg, dtype=object)
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=int)

    print("Total examples:", len(texts), "positive:", int(labels.sum()), "negative:", int((labels == 0).sum()))

    #Split 70-15-15
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=SEED, stratify=labels
    )
    dev_ratio_of_temp = 0.15 / 0.85  # so dev becomes 15% 
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_temp, y_temp, test_size=dev_ratio_of_temp, random_state=SEED, stratify=y_temp
    )

    print_split_sizes(X_train, X_dev, X_test, y_train, y_dev, y_test)

    #Manual vectorizer with classifier and on dev selection and on test eval
    print("\n Manual vectorizer with Logistic Regression ")

    #simple grid
    grid_min_df = [1, 2, 5]
    grid_max_df = [None]             
    grid_max_features = [5000, 10000, None] #sets a cap on the features
    grid_C = [0.1, 1.0, 3.0, 10.0]

    best_dev_acc = -1.0
    best_cfg = None
    best_vec = None
    best_clf = None

    for min_df in grid_min_df: #for each vectorizer config, we fit on the training data and evaluate on the dev set, keeping track of the best performing config based on dev accuracy
        for max_df in grid_max_df:
            for max_features in grid_max_features:
                vec = SimpleCountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features) #creates vectorizer
                Xtr = vec.fit_transform(X_train) #fits the vectorizer on the training data
                Xdv = vec.transform(X_dev) #development set will use the same vocab as the training set

                for C in grid_C:
                    clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear") #create logiztic regression classifier with the given C value
                    clf.fit(Xtr, y_train) #fit on the training data
                    dev_acc = clf.score(Xdv, y_dev) #getting the accuracy on the dev set

                    if dev_acc > best_dev_acc: #for storing the best dev accuracy and the configuration for it
                        best_dev_acc = dev_acc
                        best_cfg = (min_df, max_df, max_features, C, len(vec.vocab_))
                        best_vec = vec
                        best_clf = clf

    print("Best manual configuration:")
    print(f"  min_df={best_cfg[0]}, max_df={best_cfg[1]}, max_features={best_cfg[2]}, C={best_cfg[3]}, vocab_size={best_cfg[4]}")
    print("  Development accuracy:", best_dev_acc)

    # Developement report
    Xdv_best = best_vec.transform(X_dev)
    ydv_pred = best_clf.predict(Xdv_best)
    eval_and_print(y_dev, ydv_pred, "My Vectorizer - Development set Performance")

    #Test report 
    Xte_best = best_vec.transform(X_test)
    yte_pred = best_clf.predict(Xte_best)
    eval_and_print(y_test, yte_pred, "My Vectorizer - Test set Performance")

    manual_test_acc = accuracy_score(y_test, yte_pred)

    # Step 5: Comparing with sklearn CountVectorizer and teh TfidfVectorizer

    print("\n sklearn Vectorizers (CountVectorizer / TfidfVectorizer) + Logistic Regression ")
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    sk_grid = [
        ("CountVectorizer", CountVectorizer(min_df=2, max_df=0.9, max_features=20000)),
        ("TfidfVectorizer", TfidfVectorizer(min_df=2, max_df=0.9, max_features=20000)),
    ]
    C_vals = [0.1, 1.0, 3.0, 10.0] #same C values as before for a true comparison

    best2_dev_acc = -1.0
    best2_cfg = None
    best2_vec = None
    best2_clf = None

    for name, vec in sk_grid: # for each vectorizer, fit on training data and evaluate on development set
        Xtr = vec.fit_transform(X_train)
        Xdv = vec.transform(X_dev)

        for C in C_vals:
            clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
            clf.fit(Xtr, y_train)
            dev_acc = clf.score(Xdv, y_dev)

            if dev_acc > best2_dev_acc:
                best2_dev_acc = dev_acc
                best2_cfg = (name, C)
                best2_vec = vec
                best2_clf = clf

    print("Best sklearn config:")
    print(f"  vectorizer={best2_cfg[0]}, C={best2_cfg[1]}")
    print("  DEV accuracy:", best2_dev_acc)

    # Development report
    Xdv2 = best2_vec.transform(X_dev)
    ydv2_pred = best2_clf.predict(Xdv2)
    eval_and_print(y_dev, ydv2_pred, "sklearn Vectorizer - Developmenet set Performance")

    # Test report
    Xte2 = best2_vec.transform(X_test)
    yte2_pred = best2_clf.predict(Xte2)
    eval_and_print(y_test, yte2_pred, "sklearn Vectorizer - Test set Performance")

    sk_test_acc = accuracy_score(y_test, yte2_pred)


    #  comparison summary
    print("\n=== Comparison Summary ===")
    print(f"Manual vectorizer Test set accuracy : {manual_test_acc:.4f}")
    print(f"sklearn vectorizer Test set accuracy: {sk_test_acc:.4f}")

if __name__ == "__main__":
    main()