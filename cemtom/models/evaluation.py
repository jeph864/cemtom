import numpy as np


class TopicEvaluation:
    pass


def sia_npmi(topic_words, ntopics, word_doc_counts, nfiles):
    eps = 10 ** (-12)

    all_topics = []
    for k in range(ntopics):
        word_pair_counts = 0
        topic_score = []

        ntopw = len(topic_words[k])

        for i in range(ntopw - 1):
            for j in range(i + 1, ntopw):
                w1 = topic_words[k][i]
                w2 = topic_words[k][j]
                w1w2_dc = len(word_doc_counts.get(w1, set()) & word_doc_counts.get(w2, set()))
                w1_dc = len(word_doc_counts.get(w1, set()))
                w2_dc = len(word_doc_counts.get(w2, set()))
                pmi_w1w2 = np.log((w1w2_dc * nfiles) / ((w1_dc * w2_dc) + eps) + eps)
                npmi_w1w2 = pmi_w1w2 / (- np.log((w1w2_dc) / nfiles + eps))
                topic_score.append(npmi_w1w2)

        all_topics.append(np.mean(topic_score))

    for k in range(ntopics):
        print(np.around(all_topics[k], 5), " ".join(topic_words[k]))

    avg_score = np.around(np.mean(all_topics), 5)
    # print(f"\nAverage NPMI for {ntopics} topics: {avg_score}")

    return avg_score
