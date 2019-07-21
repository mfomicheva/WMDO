import os

from WMDO.tests.utils import create_vectors_file
from WMDO.wmdo import load_wv
from WMDO.wmdo import wmdo


def test_computes_wmdo_same_languge():
    vectors_file = create_vectors_file()
    vectors = load_wv(vectors_file, binned=False)
    os.remove(vectors_file)
    test = 'test .'
    reference = 'test .'
    score = wmdo(vectors, test, reference, delta=0.18, alpha=0.10)
    assert score == float('inf')
    assert score


def test_computes_wmdo_cross_lingual():
    vectors_file = create_vectors_file()
    vectors = load_wv(vectors_file, language='en', binned=False)
    vectors = load_wv(vectors_file, existing=vectors, language='de', binned=False)
    os.remove(vectors_file)
    test = 'test'
    reference = 'test'
    score = wmdo(vectors, reference, test, ref_lang='en', cand_lang='de', delta=0.18, alpha=0.10)
    assert score == float('inf')
    assert score
