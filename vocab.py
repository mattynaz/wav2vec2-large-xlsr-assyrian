import re

vocab = {
    ' ': 0,
    '+': 1,
    '-': 2,
    'a': 3,
    'b': 4,
    'c': 5,
    'd': 6,
    'e': 7,
    'f': 8,
    'h': 9,
    'i': 10,
    'j': 11,
    'k': 12,
    'l': 13,
    'm': 14,
    'n': 15,
    'o': 16,
    'p': 17,
    'r': 18,
    's': 19,
    't': 20,
    'u': 21,
    'v': 22,
    'x': 23,
    'y': 24,
    'z': 25,
    'č': 26,
    'ġ': 27,
    'š': 28,
    'ž': 29,
    'ə': 30,
    'ɟ': 31,
    'ʾ': 32,
    'α': 33,  # k̭
    'β': 34,  # p̂
    'γ': 35,  # č̭
    'δ': 36,  # c̭
    'ṱ': 37,
    # '[UNK]': 38,
    # '[PAD]': 39
}

replace_res = [
    # vowel simplification
    ('À', 'A'),
    ('à', 'a'),
    ('á', 'a'),
    ('è', 'e'),
    ('é', 'e'),
    ('ì', 'i'),
    ('í', 'i'),
    ('ò', 'o'),
    ('ó', 'o'),
    ('ù', 'u'),
    ('ú', 'u'),
    ('ā', 'a'),
    ('ă', 'a'),
    ('ä', 'a'),
    ('ē', 'e'),
    ('ī', 'i'),
    ('ō', 'o'),
    ('ū', 'u'),
    ('ǝ', 'ə'),
    ('ɑ', 'a'),
    ('ḕ', 'e'),
    ('ḗ', 'e'),
    ('ṑ', 'o'),
    ('ṓ', 'o'),
    ('ắ', 'a'),
    ('ằ', 'a'),

    # parse mistakes
    ('kə̭', 'k̭ə'),
    ('ka̭', 'k̭a'),
    ('kr̭', 'k̭r'),
    (' -', '-'),

    # misc
    ('=', '-'),
    ('꞊', '-'),
]

ligature_res = [
    ('ṱ', 'ṱ'),
    ('k̭', 'α'),
    ('p̂', 'β'),
    ('č̭', 'γ'),
    ('č̭', 'γ'),
    ('c̭', 'δ'),
]

ignore_re = r'[…():|ˈ!?,.\d̄̀́]|(?:P(.|\n)+?P)|(?:E(.|\n)+?E)'

strip_res = [
    ('\s+', ' '),
    ('^\s+|\s+$', ''),
]


def normalize(text: str, panic: bool = True) -> str:
    '''
    Normalize transcription of `text`. If `panic`,
    will panic
    '''
    # Initial changes: simplify vowels, remove parse mistakes,
    # consolidate, ligatures, etc. 
    for pattern, repl in replace_res + ligature_res:
        text = re.sub(pattern, repl, text)

    # Ignore punctuation, numbers, and foreign segments
    text = re.sub(ignore_re, '', text)

    # Clean whitespace
    for pattern, repl in strip_res:
        text = re.sub(pattern, repl, text)

    # Make lowercase
    text = text.lower()
    
    if panic:
        unknowns = set(text) - set(vocab.keys())
        if len(unknowns) != 0:
            raise Exception(f'Cannot normalize. Unknown characters: {unknowns}')

    return text


def restore(text: str) -> str:
    '''
    Restore the transcription from the normalized
    alphabet to standard alphabet.
    '''

    # Restore ligatures
    for repl, pattern in ligature_res:
        text = re.sub(pattern, repl, text)
    
    return text
