import numpy as np
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from pyemd import emd_with_flow
from nltk import word_tokenize
from gensim.models import KeyedVectors
from bisect import bisect_left


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

        return chunks / matched_unigrams
    except IndexError:
        return 0


def get_input_words(text, language):
    return ['{}{}'.format(language, w.lower()) for w in word_tokenize(text.replace('-', ' '))]


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

    ref_list = get_input_words(ref, ref_lang)
    cand_list = get_input_words(cand, cand_lang)

    ref = ' '.join(ref_list)
    cand = ' '.join(cand_list)

    vc = CountVectorizer().fit(ref_list + cand_list)

    v_obj, v_cap = vc.transform([ref, cand])

    v_obj = v_obj.toarray().ravel()
    v_cap = v_cap.toarray().ravel()

    dim = wvvecs[ref_lang].vector_size

    # need to deal with missing words
    wvoc = []
    missing = {}

    for w in vc.get_feature_names():
        lang = w[:2]
        word = w[2:]
        try:
            if word in wvvecs[lang]:
                wvoc.append(wvvecs[lang][word])
            else:
                if word not in missing:
                    missing[word] = np.zeros(dim)
                wvoc.append(missing[word])
        except KeyError:
            # tokenization mismatch, can be partially fixed by using CountVectorizer(token_pattern=r'(?u)\b[A-Za-z\-\'][A-Za-z\-\']+\b')
            sys.stderr('Different tokenization. Skipping word {}'.format(word))
            if word not in missing:
                sys.stderr('Word {} is missing'.format(word))
                missing[word] = np.zeros(dim)
            wvoc.append(missing[word])

    distance_matrix = cosine_distances(wvoc)

    if np.sum(distance_matrix) == 0.0:
        return float('inf')

    v_obj = v_obj.astype(np.double)
    v_cap = v_cap.astype(np.double)

    v_obj /= v_obj.sum()
    v_cap /= v_cap.sum()

    distance_matrix = distance_matrix.astype(np.double)
    (wmd, flow) = emd_with_flow(v_obj, v_cap, distance_matrix)

    # adding penalty
    ratio = fragmentation(ref_list, cand_list, vc, flow)
    if ratio > 1:
        ratio = 1
    penalty = delta * ratio

    # missing words penalty
    missingwords = 0
    for w in cand_list:
        if w not in wvvecs:
            missingwords += 1
    missingratio = missingwords / len(cand_list)
    missing = alpha * missingratio
    
    penalty += missing

    wmd += penalty
    
    return wmd
