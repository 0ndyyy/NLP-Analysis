import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

#Quality variables
minimum_required_description = 3
min_topics = 2
max_topics = 10

bow_min_df = 2
bow_max_df = 0.8

tfidf_min_df = 2
tfidf_max_df = 0.8

#Global variables
data_path = "public_tickets_final.csv"
signature_automatic_tickets = ["[Review]", "[Wartung]", "[Update]"]
pattern = "|".join(signature_automatic_tickets)
german_stopwords = set(stopwords.words("german"))
english_stopwords = set(stopwords.words("english"))
custom_stopwords = ["bitte", "hallo", "danke", "guten", "tag", "it", "arbeiten", "mittlerweile", "kollege", "e", "review", "wartung", "fritz", "winter", "fritzwinter", "sc", "de", "dank", "grüße", "freundlich", "wurde", "frau", "herr", "fwde", "com", "eisengießerei", "www", "eur", "aw", "kg", "lo", "us", "fw", "na", "lieb", "external", "automatisch", "winterianer", "winterianerin", "anfrage", "anzeigen", "benötigen", "links", "score", "results", "sender", "lieber", "ansprechpartner", "rückfrage", "vieler", "betreff", "gruß", "neu", "detail", "lieb", "generiert", "lieber", "weit", "finden", "funktionieren", "mal", "bzw", "bei", "mehr", "mal", "vieler", "anbei", "immer", "ca", "sobald", "seit", "weit", "km", "möglich", "wg", "generierte", "liebe", "weit", "vieler", "benötigen", "lc", "morgen", "benötigen", "funktionieren", "neu", "vieler", "eu", "fd", "safelink", "https", "uri", "url", "ekg", "zeichen", "lang", "fd", "re", "ih", "cc", "rückfrage", "extern"]
all_stopwords = german_stopwords.union(custom_stopwords).union(english_stopwords)
lemmas = spacy.load("de_core_news_lg")

#Returns DataFrame with columns defined
def load_csv(path):
    df = pd.read_csv(path, skiprows=2, sep=";")
    #Combine Subject + Description
    df["complete_text"] = (
        df["Subject"].fillna("") + " " +
        df["Description"].fillna("")
    )
    print("successfully loaded .csv")
    return df

#Returns DataFrame without unnessecary tickets
def validate_tickets(df):
    tickets_before_validation = len(df)
    
    #Strip empty Tickets
    df = df[df["Description"].notna()]
    df = df[df["Description"].str.strip() != ""]

    #Strip Tickets with no description or description < 5 words
    df = df[df["Description"].str.split().str.len() > minimum_required_description]

    #Double Tickets
    df = df.drop_duplicates(subset=["Description"])

    #Strip automatically generated tickets
    automatic_ticket_mask = df["Description"].apply(
        lambda text: any(
            signature.lower() in text.lower()
            for signature in signature_automatic_tickets
        )
    )
    df = df[~automatic_ticket_mask]
    removed_tickets = tickets_before_validation - len(df)
    
    print("Removed Tickets through validation:", removed_tickets)
    return df

#Returns tokenized tickets
def preprocessing(df):

    #NER to remove persons
    df["postNER_complete_text"] = df["complete_text"].apply(
        lambda text: " ".join(
            token.text
            for token in lemmas(text)
            if token.ent_type_ != "PER"
        )
    )

    #Lowercase
    df["lower_complete_text"] = df["postNER_complete_text"].str.lower()

    #Tokenize and remove punctation
    tokenizer = nltk.RegexpTokenizer(r"[A-Za-zÄÖÜäöüß]+")
    df["tokens"] = df["lower_complete_text"].apply(tokenizer.tokenize)

    #Remove stopwords
    df["cleansed_tokens"] = df["tokens"].apply(
    lambda tokens: [
        word for word in tokens
        if word not in all_stopwords
        ]
    )  

    #Lemmantisation
    df["lemmas"] = df["cleansed_tokens"].apply(
        lambda tokens: [
            token.lemma_
            for token in lemmas(" ".join(tokens))
        ]
    )

    print("successfully preproccessed")
    return df

#Returns ticket-word matrix 
def create_bow(df):

    joined_lemmas = df["lemmas"].apply(" ".join)
    vectorizer = CountVectorizer(min_df= bow_min_df, max_df= bow_max_df)
    bow_matrix = vectorizer.fit_transform(joined_lemmas)

    print("successfully created BoW")
    return bow_matrix, vectorizer

def find_optimal_topic_count_lda(df, bow_matrix, vectorizer, min_topics, max_topics):

    #restrict documents to lda vocabulary
    texts = df["lemmas"].tolist()
    dictionary = Dictionary(texts)

    results = []
    best_score = -1
    best_topic_count = None

    vocabulary = set(vectorizer.get_feature_names_out())

    for topic_count in range(min_topics, max_topics + 1):

        print(f"Testing {topic_count} topics...")

        #train lda
        lda = LatentDirichletAllocation(n_components=topic_count, learning_method="online", random_state=42)
        lda.fit(bow_matrix)

        #extract top words
        topics = []

        for topic in lda.components_:
            top_indices = topic.argsort()[::-1][:10]
            top_words = [vectorizer.get_feature_names_out()[index] for index in top_indices]
            topics.append(top_words)

        #calculate coherence
        coherence_model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence="c_v")
        coherence_score = coherence_model.get_coherence()
        results.append({"topic_count": topic_count, "coherence": coherence_score})

        #save best model
        if coherence_score > best_score:
            best_score = coherence_score
            best_topic_count = topic_count

    results = pd.DataFrame(results)

    print("\nOptimal number of topics:")
    print(best_topic_count)

    print(f"Coherence Score: {best_score:.4f}")

    return best_topic_count, results

#Trains the LDA model based on created BoW. Returns LDA and Topics
def train_lda(bow_matrix, topics_to_discover):
    #configure lda to make output reproduceable
    lda = LatentDirichletAllocation(n_components=topics_to_discover, learning_method="online", random_state=42)
    lda.fit(bow_matrix)

    #discover topic per ticket
    ticket_topics = lda.transform(bow_matrix)

    print("successfully trained LDA")
    return lda, ticket_topics


#Returns top topics, top probabilities and most ambigous tickets for later visualization
def lda_analyze_topics(df, ticket_topics, lda_model, vectorizer):

    #Primary topic per ticket
    df["lda_primary_topic"] = ticket_topics.argmax(axis=1)

    #Probability of primary topic
    df["lda_topic_probability"] = ticket_topics.max(axis=1)

    #Topic distribution
    lda_topic_distribution = df["lda_primary_topic"].value_counts().sort_index().rename_axis("topic").reset_index(name="ticket_count")

    #Representative tickets
    lda_topic_representative_results = []

    for topic_id in range(ticket_topics.shape[1]):
        topic_probabilities = ticket_topics[:, topic_id]

        top_indices = (topic_probabilities.argsort()[::-1][:5])

        for index in top_indices:
            lda_topic_representative_results.append({
                "topic": topic_id,
                "probability": topic_probabilities[index],
                "request_id": df.iloc[index]["Request ID"],
                "subject": df.iloc[index]["Subject"]
            })

    lda_topic_representative_tickets = pd.DataFrame(lda_topic_representative_results)

    #Ambiguous tickets
    ambiguous_results = []

    for topic_id in range(ticket_topics.shape[1]):
        topic_tickets = df[df["lda_primary_topic"] == topic_id]

        ambiguous = (topic_tickets.sort_values("lda_topic_probability").head(5))

        for _, ticket in ambiguous.iterrows():
            ambiguous_results.append({
                "topic": topic_id,
                "probability": ticket["lda_topic_probability"],
                "request_id": ticket["Request ID"],
                "subject": ticket["Subject"]
            })

    lda_topic_ambiguous_tickets = pd.DataFrame(ambiguous_results)

    #Top words per topic
    vocabulary = vectorizer.get_feature_names_out()

    top_word_results = []

    for topic_id, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[::-1][:10]

        for index in top_indices:
            top_word_results.append({
                "topic": topic_id,
                "word": vocabulary[index],
                "weight": topic[index]
            })

    lda_topic_top_words = pd.DataFrame(top_word_results)

    print("successfully analyzed LDA")
    return lda_topic_distribution, lda_topic_representative_tickets, lda_topic_ambiguous_tickets

#Returns most important words per ticket and vectorizer
def create_tfidf(df):

    joined_lemmas = df["lemmas"].apply(" ".join)
    vectorizer = TfidfVectorizer(min_df = tfidf_min_df, max_df = tfidf_max_df)
    tfidf_matrix = vectorizer.fit_transform(joined_lemmas)

    print("successfully created TF-IDF")
    return tfidf_matrix, vectorizer

#Analyzes most important words per Topic and prints them
def analyze_tfidf(df, tfidf_matrix, tfidf_vectorizer):

    vocabulary = tfidf_vectorizer.get_feature_names_out()

    results = []

    for topic_id in sorted(df["lda_primary_topic"].unique()):

        topic_indices = [i for i, topic in enumerate(df["lda_primary_topic"])
            if topic == topic_id
        ]

        topic_tfidf = tfidf_matrix[topic_indices]

        mean_tfidf = topic_tfidf.mean(axis=0).A1

        top_indices = mean_tfidf.argsort()[::-1][:10]

        for index in top_indices:
            results.append({
                "topic": topic_id,
                "word": vocabulary[index],
                "weight": mean_tfidf[index]
            })

    print("successfully analyzed TF-IDF")
    return pd.DataFrame(results)

#Train LSA model on TF-IDF
def train_lsa(tfidf_matrix, topics_to_discover):

    lsa = TruncatedSVD(n_components=topics_to_discover, random_state=42)

    lsa.fit(tfidf_matrix)

    #Discover latent topics per ticket
    ticket_topics = lsa.transform(tfidf_matrix)

    print("successfully trained LSA")
    return lsa, ticket_topics

#saves the most important words per LSA topic in df
def create_lsa_top_words(lsa_model, vectorizer, n_words=10):

    vocabulary = vectorizer.get_feature_names_out()
    results = []

    for component_id, component in enumerate(lsa_model.components_):
        top_indices = component.argsort()[::-1][:n_words]

        for index in top_indices:
            results.append({
                "component": component_id,
                "word": vocabulary[index],
                "weight": component[index]
            })

    return pd.DataFrame(results)

def create_lda_visualization_data(df):

    topic_distribution = df["lda_primary_topic"].value_counts().sort_index()

    return pd.DataFrame({
        "topic": topic_distribution.index,
        "ticket_count": topic_distribution.values
    })

def create_lsa_visualization_data(lsa_model):

    lsa_results = pd.DataFrame({
        "component": range(
            len(lsa_model.explained_variance_ratio_)
        ),
        "explained_variance":
            lsa_model.explained_variance_ratio_
    })

    lsa_results["cumulative_variance"] = lsa_results["explained_variance"].cumsum()
    
    return lsa_results

def plot_coherence(coherence_results, optimal_topic_count):

    plt.figure(figsize=(10, 6))

    plt.plot(
        coherence_results["topic_count"],
        coherence_results["coherence"],
        marker="o"
    )

    plt.xlabel("Anzahl Topics")
    plt.ylabel("Coherence Score (c_v)")
    plt.title("Bestimmung der optimalen LDA-Topic-Anzahl")

    plt.xticks(coherence_results["topic_count"])

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    #User selection
    print(
        f"\nAutomatically determined optimal topic count: " f"{optimal_topic_count}"
    )

    while True:

        user_input = input(
            f"Enter topic count ({min_topics}-{max_topics}) "
            f"or press Enter to use {optimal_topic_count}: "
        )

        #Use optimum if user returns empty string
        if user_input.strip() == "":
            return optimal_topic_count

        try:
            selected_topic_count = int(user_input)

            if min_topics <= selected_topic_count <= max_topics:
                return selected_topic_count

            print(
                f"Please enter a value between " f"{min_topics} and {max_topics}."
            )

        except ValueError:
            print("Please enter a valid integer")

def plot_lda_topic_distribution(topic_results):

    fig, ax = plt.subplots(figsize=(12, 5))

    bars = ax.bar(topic_results["topic"], topic_results["ticket_count"])

    ax.set_title("Verteilung Tickets / LDA-Topics")
    ax.set_xlabel("LDA-Topic")
    ax.set_ylabel("Anzahl Tickets")
    ax.set_xticks(topic_results["topic"])

    #values over bars
    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.show()

def plot_lda_topic_words(tfidf_topic_words):

    topics = sorted(tfidf_topic_words["topic"].unique())

    ncols = 3
    nrows = math.ceil(len(topics) / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(15, nrows * 4)
    )

    axes = axes.flatten()

    for plot_index, topic_id in enumerate(topics):

        ax = axes[plot_index]
        topic_data = tfidf_topic_words[tfidf_topic_words["topic"] == topic_id].sort_values("weight")
        ax.barh(topic_data["word"], topic_data["weight"])
        ax.set_title(f"LDA Topic {topic_id}")
        ax.set_xlabel("Durchschnittlicher TF-IDF-Wert")
        ax.set_ylabel("")

    #remove uneccessary subplots
    for ax in axes[len(topics):]:
        ax.remove()

    fig.suptitle("Charakteristische Wörter der LDA-Topics", fontsize=16)

    plt.tight_layout()
    plt.show()

def plot_lda_clusters(lda_ticket_topics):

    pca = PCA(n_components=2)
    lda_2d = pca.fit_transform(lda_ticket_topics)
    primary_topics = lda_ticket_topics.argmax(axis=1)
    topic_probability = lda_ticket_topics.max(axis=1)

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        lda_2d[:, 0],
        lda_2d[:, 1],
        c=primary_topics,
        s=topic_probability * 100,
        alpha=0.7
    )

    plt.xlabel(f"PCA-Komponente 1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PCA-Komponente 2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title("LDA-Topic-Struktur der Tickets")
    plt.colorbar(scatter,label="Primäres LDA-Topic")

    plt.tight_layout()
    plt.show()

def plot_lsa_topic_words(lsa_top_words):

    components = sorted(lsa_top_words["component"].unique())

    ncols = 4
    nrows = math.ceil(len(components) / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(16, nrows * 4)
    )

    axes = axes.flatten()

    for plot_index, component_id in enumerate(components):

        ax = axes[plot_index]
        component_data = (lsa_top_words[lsa_top_words["component"] == component_id].sort_values("weight"))
        ax.barh(component_data["word"], component_data["weight"])

        ax.set_title(f"LSA-Komponente {component_id}")
        ax.set_xlabel("Gewicht")

    #remove not needed axes
    for ax in axes[len(components):]:
        ax.remove()

    plt.suptitle("Charakteristische Wörter der LSA-Komponenten", fontsize=16)

    plt.tight_layout()
    plt.show()

def plot_lsa_document_space(lsa_ticket_topics):

    plt.figure(figsize=(12, 8))

    plt.scatter(
        lsa_ticket_topics[:, 0],
        lsa_ticket_topics[:, 1],
        alpha=0.6
    )

    plt.xlabel("LSA-Komponente 1")
    plt.ylabel("LSA-Komponente 2")
    plt.title("LSA-Struktur der Tickets")

    plt.tight_layout()
    plt.show()


#Execution Path
if __name__ == "__main__":
    #Preprocessing
    df = load_csv(data_path)
    df_validated = validate_tickets(df)
    df_preprocessed = preprocessing(df_validated)

    #Pipeline 1: BoW -> LDA -> TF-IDF (= in Pipeline 2)
    bow_matrix, bow_vectorizer = create_bow(df_preprocessed)
    #find best topic count for both approaches (as lda is favored, it is calculated for lda and only applied to lsa)
    topics_to_discover, coherence_results = find_optimal_topic_count_lda(df_preprocessed, bow_matrix, bow_vectorizer, min_topics=min_topics, max_topics=max_topics)
    topics_to_discover = plot_coherence(coherence_results, topics_to_discover)
    #train lda based on topics (rendunand, as the model was trained above, but I wanted to capusle the function)
    lda_model, lda_ticket_topics = train_lda(bow_matrix, topics_to_discover)
    lda_topic_distribution, lda_topic_representative_tickets, lda_topic_ambiguous_tickets = lda_analyze_topics(df_preprocessed, lda_ticket_topics, lda_model, bow_vectorizer)

    #Pipeline 2 TF-IDF -> LSA
    tfidf, tfidf_vectorizer = create_tfidf(df_preprocessed)
    lsa_model, lsa_ticket_topics = train_lsa(tfidf, topics_to_discover)
    tfidf_topic_words = analyze_tfidf(df_preprocessed, tfidf, tfidf_vectorizer)

    #Visualization LDA
    lda_topic_results = create_lda_visualization_data(df_preprocessed)
    plot_lda_topic_distribution(lda_topic_results)
    plot_lda_topic_words(tfidf_topic_words)
    plot_lda_clusters(lda_ticket_topics)

    #Visualization LSA
    lsa_results = create_lsa_visualization_data(lsa_model)
    lsa_top_words = create_lsa_top_words(lsa_model, tfidf_vectorizer)
    plot_lsa_topic_words(lsa_top_words)
    plot_lsa_document_space(lsa_ticket_topics)




