import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from pyemd import emd_with_flow
from nltk import word_tokenize
from gensim.models import KeyedVectors
from bisect import bisect_left


def create_vocabulary(count_vectorizer, vectors, dim, ref_list, cand_list, ref_lang, cand_lang):
    wvoc = []
    missing = {}
    for word in count_vectorizer.get_feature_names():
        if word in ref_list:
            if word in vectors[ref_lang]:
                wvoc.append(vectors[ref_lang][word])
            else:
                if word not in missing:
                    missing[word] = np.zeros(dim)
                wvoc.append(missing[word])
        else:
            if word in vectors[cand_lang]:
                wvoc.append(vectors[cand_lang][word])
            else:
                if word not in missing:
                    missing[word] = np.zeros(dim)
                wvoc.append(missing[word])
    return wvoc, missing


def load_wv(path, language='en', existing=None, binned=False):
    W = KeyedVectors.load_word2vec_format(path, binary=binned)
    W.init_sims(replace=True)
    result = existing if existing else {}
    assert not language in result
    result[language] = W
    return result


def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def fragmentation(ref_list, cand_list, vc, flow):
    try:
        chunks = 1
        matched_unigrams = 0

        current = -1
        for i, w in enumerate(ref_list):
            if w in vc.get_feature_names():
                index = vc.get_feature_names().index(w)
                flow_from_w = flow[index]
                highest_flow = max(flow_from_w)

                if not highest_flow == 0:
                    feature_names_matched_indices = [
                        i for i, x in enumerate(flow_from_w) if x == highest_flow]
                    matched_words = [vc.get_feature_names()[i]
                                     for i in feature_names_matched_indices]

                    # check cases where word doesn't map to anything.
                    matched_indices = []
                    for m in matched_words:
                        occurrences = []
                        for i, x in enumerate(cand_list):
                            if x == m:
                                occurrences.append(i)
                        matched_indices.append(
                            takeClosest(occurrences, current))
                    matched_index = takeClosest(matched_indices, current)

                    if not current + 1 == matched_index:
                        chunks += 1
                    current = matched_index
                    matched_unigrams += 1

        try:
            return chunks / matched_unigrams
        except ZeroDivisionError:
            return 0
    except IndexError:
        return 0


def get_input_words(text):
    return [w.lower() for w in word_tokenize(text)]


def wmdo(wvvecs, ref, cand, ref_lang='en', cand_lang='en', delta=0.18, alpha=0.1):
    '''
    wvvecs: word vectors -- retrieved from load_wv method
    ref: reference translation
    cand: candidate translation
    missing: missing word dictionary -- initialise as {}
    dim: word vector dimension
    delta: weight of fragmentation penalty
    alpha: weight of missing word penalty
    '''

    ref_list = get_input_words(ref)
    cand_list = get_input_words(cand)

    ref = ' '.join(ref_list)
    cand = ' '.join(cand_list)

    common_vectorizer = CountVectorizer().fit(ref_list + cand_list)

    ref_count_vector, cand_count_vector = common_vectorizer.transform([ref, cand])

    ref_count_vector = ref_count_vector.toarray().ravel()
    cand_count_vector = cand_count_vector.toarray().ravel()

    dim = wvvecs[ref_lang].vector_size

    wvoc, missing = create_vocabulary(common_vectorizer, wvvecs, dim, ref_list, cand_list, ref_lang, cand_lang)

    distance_matrix = cosine_distances(wvoc)
    vocab_words = common_vectorizer.get_feature_names()
    for cand_word_idx, count in enumerate(cand_count_vector):
        if count > 0:
            most_similar_ref_indexes = np.argsort(distance_matrix[cand_word_idx])
            for ref_word_index in most_similar_ref_indexes[1:]:
                if ref_count_vector[ref_word_index] > 0:
                    print('{}: {}'.format(vocab_words[cand_word_idx], vocab_words[ref_word_index]))
                    break

    if np.sum(distance_matrix) == 0.0:
        return 0., {}
        #return float('inf')

    ref_count_vector = ref_count_vector.astype(np.double)
    cand_count_vector = cand_count_vector.astype(np.double)

    ref_count_vector /= ref_count_vector.sum()
    cand_count_vector /= cand_count_vector.sum()

    distance_matrix = distance_matrix.astype(np.double)
    (wmd, flow) = emd_with_flow(ref_count_vector, cand_count_vector, distance_matrix)

    return wmd, {}

    # adding penalty
    ratio = fragmentation(ref_list, cand_list, common_vectorizer, flow)
    if ratio > 1:
        ratio = 1
    penalty = delta * ratio

    # missing words penalty
    missingwords = 0
    for w in cand_list:
        if w not in wvvecs:
            missingwords += 1
    missingratio = missingwords / len(cand_list)
    missing_penalty = alpha * missingratio
    
    penalty += missing_penalty

    wmd += penalty
    
    return wmd, missing
