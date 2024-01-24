from cemtom.dataset import fetch_dataset
from cemtom.preprocessing import Preprocessor as Pipe
from cemtom.embedder import BertVocabEmbedder, get_word_embedding_model
from cemtom.models import Sia
import argparse


def main(args_):
    dataset = fetch_dataset(name=args_.dataset, remove=("headers", "footers", "quotes"))
    token_dict = {
        "doc_start_token": '<s>',
        "doc_end_token": '</s>',
        "unk_token": '<unk>',
        "email_token": '<email>',
        "url_token": '<url>',
        "number_token": '<number>',
        "alpha_num_token": '<alpha_num>'
    }
    vocabulary = None
    vocab_embeddings = None
    embedding_model = None
    model = None
    if args_.vocab_embedding == "bert":
        embedding_model = BertVocabEmbedder.load_embeddings(args_.vocab_embeddings_path)
        vocabulary = embedding_model.vocab
        vocab_embeddings = embedding_model.embeddings

    pipe = Pipe(stopwords_list="english", remove_spacy_stopwords=False,
                token_dict=token_dict, use_spacy_tokenizer=True, min_df=5,
                max_df=0.80, vocabulary=vocabulary)
    data = pipe.preprocess(None, dataset=dataset)
    train_corpus, test_corpus = data.get_partitioned()
    corpus = train_corpus + test_corpus
    vocabulary = pipe.vectorizer.get_feature_names_out()

    if args_.vocab_embedding == "fasttext":
        embedding_model = get_word_embedding_model(name="fasttext", path=args_.vocab_embeddings_path)
        vocab_embeddings = embedding_model.embed(vocabulary)
    if args_.model == "sia":
        model = Sia(vocab=vocabulary, embedding_model_name=args_.vocab_embedding, vectorizer=pipe.vectorizer,
                    nr_dimensions=args_.red_dimensions,
                    nr_topics=args_.nr_topics,
                    weighting=args_.weighting,
                    rerank=args_.rerank,
                    word2idx=embedding_model.word2idx
                    )
        model.fit_transform(corpus, embeddings=vocab_embeddings)
        model.save_topics(path=f"sia_{args_.dataset}_{args_.vocab_embedding}_topics.json")
    if args_.model == "bertopic":
        pass
    if args_.model == "CTM":
        pass
    else:
        raise ValueError("No valid topic model provided")

    if args_.topic_eval:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--vocab_embedding", type=str, default="bert", choices=["bert", "fasttext", "glove"])
    parser.add_argument("--vocab_embedding_path", type=str, )
    parser.add_argument("--model", type=str, default="sia", help="the topic model")

    parser.add_argument("--clustering_algo", type=str, required=True,
                        choices=["KMeans", "SPKMeans", "GMM", "KMedoids", "Agglo", "DBSCAN", "Spectral", "VMFM"])

    parser.add_argument("--topics_file", type=str, default="topics.json", help="topics file")

    parser.add_argument('--use_dims', type=int)
    parser.add_argument('--num_topics', type=int, default=20)
    parser.add_argument("--doc_info", type=str, choices=["SVD", "DUP", "WGT", "robust", "logtfdf"])
    parser.add_argument("--rerank", type=str, choices=["tf", "tfidf", "tfdf", "graph"])

    parser.add_argument("--dataset", type=str, default="20Newsgroup", choices=["20Newsgroup", "children", "reuters"])

    parser.add_argument("--stopwords", type=str, help="Path to stopwords")

    parser.add_argument("--vocab", required=True, type=str, nargs='+', default=[])
    parser.add_argument("--scale", type=str, required=False)

    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':
    args = parse_args()
    main(args)
