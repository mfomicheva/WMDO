import tempfile


def create_vectors_file():
    header = '4 5'
    lines = [
        'test -0.0462662 -0.0162351 -0.0123162 0.0222278 -0.0730516',
        'cat -0.0462662 -0.0162351 -0.0123162 0.0222278 -0.0730516',
        'dog -0.0462662 -0.0162351 -0.0123162 0.0222278 -0.0730516',
        'boy -0.0462662 -0.0162351 -0.0123162 0.0222278 -0.0730516'
    ]
    _, vectors_file = tempfile.mkstemp()
    with open(vectors_file, 'w') as o:
        o.write(header + '\n')
        for line in lines:
            o.write(line + '\n')
    o.close()
    return vectors_file
