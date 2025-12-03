

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    max_features=20000
)


X_train_vec = vectorizer.fit_transform(X_train)
clf = LogisticRegression(max_iter=2000, n_jobs=-1)
clf.fit(X_train_vec, y_train)


URGENCY_WORDS  = ["immediately","urgent","now","today"]
AUTHORITY_WORDS = ["bank","security team","compliance","irs"]
REWARD_WORDS    = ["profit","bonus","jackpot"]
FEAR_WORDS      = ["suspended","blocked","penalty"]


def score_keyword_group(text, keywords):
    t = text.lower()
    hits = sum(1 for kw in keywords if kw in t)
    return min(hits, 3) / 3.0


def compute_seri(row):
    R_b = (row["urgency_score"] + row["authority_score"] +
           row["reward_score"] + row["fear_score"]) / 4.0

    R_c = row["p_scam"]
    U   = 1.0 - row["max_class_prob"]

    seri_raw = 0.35*R_b + 0.45*R_c + 0.20*U
    return 100 * np.clip(seri_raw, 0, 1)
