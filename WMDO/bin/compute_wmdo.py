import argparse
import sys

from scipy.stats import pearsonr

from WMDO.wmdo import load_wv
from WMDO.wmdo import wmdo


def read_file(path):
    def _normalize(line):
        return ' '.join(line.split())
    return [_normalize(line) for line in open(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference', required=True)
    parser.add_argument('-t', '--translation', required=True)
    parser.add_argument('-l', '--languages', nargs='+', required=True)
    parser.add_argument('-v', '--vectors', nargs='+', required=True)
    parser.add_argument('-o', '--output', required=False, default=None)
    parser.add_argument('-j', '--judgments', required=False, default=None)
    parser.add_argument('-b', '--binary', required=False, default=False, action='store_true')
    args = parser.parse_args()

    reference_lines = read_file(args.reference)
    translation_lines = read_file(args.translation)
    assert len(reference_lines) == len(translation_lines)
    languages = args.languages
    vector_files = args.vectors
    assert len(vector_files) == len(languages)

    print('Loading vectors...')
    vectors = load_wv(vector_files[0], language=languages[0], binned=args.binary)
    if len(languages) > 1:
        vectors = load_wv(vector_files[1], language=languages[1], existing=vectors, binned=args.binary)

    print('Computing wmdo...')
    scores = []
    missing = []
    out = open(args.output, 'w') if args.output else sys.stdout
    for ref, trans in zip(reference_lines, translation_lines):
        ref_lang = languages[0]
        if len(languages) > 1:
            cand_lang = languages[1]
        else:
            cand_lang = ref_lang
        score, misses = wmdo(vectors, ref, trans, ref_lang=ref_lang, cand_lang=cand_lang)
        scores.append(score)
        missing.append(len(misses))
        out.write('{}\n'.format(score))

    if args.judgments:
        judgments = [float(l.strip()) for l in open(args.judgments)]
        out_scores, out_judges = [], []
        for i, (j, s) in enumerate(zip(scores, judgments)):
            if missing[i] == 0:
                out_scores.append(s)
                out_judges.append(j)

        print('corr: {}; pval: {}'.format(*pearsonr(out_scores, out_judges)))


if __name__ == '__main__':
    main()
