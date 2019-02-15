import bin.module.util as util


TOKEN = util.general.SettingContainer(
    START = '_startTk',
    END = '_endTk'
)
MODEL = util.general.SettingContainer(
    INPUTS_ENCODER = 'inputs_encoder',
    INPUTS_DECODER = 'inputs_decoder',
    EMB_ENCODER = 'Emb_Encoder',
    EMB_DECODER = 'Emb_Decoder',
    LSTM_ENCODER = 'LSTM_Encoder',
    LSTM_DECODER = 'LSTM_Decoder',
    OUTPUTS = 'outputs'
)
NAME = util.general.SettingContainer(
    MODEL_ENCODERDECODER = 'model_encoder-decoder',
    MODEL_SENTIMENT = 'model_sentiment'
)
SAMPLE_AT = """
NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.
Thanks to a hands-on guide introducing programming fundamentals alongside topics in computational linguistics, plus comprehensive API documentation, NLTK is suitable for linguists, engineers, students, educators, researchers, and industry users alike. NLTK is available for Windows, Mac OS X, and Linux. Best of all, NLTK is a free, open source, community-driven project.
NLTK has been called “a wonderful tool for teaching, and working in, computational linguistics using Python,” and “an amazing library to play with natural language.”
Natural Language Processing with Python provides a practical introduction to programming for language processing. Written by the creators of NLTK, it guides the reader through the fundamentals of writing Python programs, working with corpora, categorizing text, analyzing linguistic structure, and more. The online version of the book has been been updated for Python 3 and NLTK 3. (The original Python 2 version is still available at http://nltk.org/book_1ed.)
"""