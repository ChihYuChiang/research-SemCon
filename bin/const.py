class TOKEN():
    START = '_startTk'
    END = '_endTk'

class MODEL():
    INPUTS_ENCODER = 'inputs_encoder'
    INPUTS_DECODER = 'inputs_decoder'
    EMB_ENCODER = 'Emb_Encoder'
    EMB_DECODER = 'Emb_Decoder'
    LSTM_ENCODER = 'LSTM_Encoder'
    LSTM_DECODER = 'LSTM_Decoder'
    OUTPUTS = 'outputs'

class NAME():
    MODEL_ENCODERDECODER = 'model_encoder-decoder'
    MODEL_SENTIMENT = 'model_sentiment'

class SAMPLE_AT():
    A = """
        NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.
        Thanks to a hands-on guide introducing programming fundamentals alongside topics in computational linguistics, plus comprehensive API documentation, NLTK is suitable for linguists, engineers, students, educators, researchers, and industry users alike. NLTK is available for Windows, Mac OS X, and Linux. Best of all, NLTK is a free, open source, community-driven project.
        NLTK has been called “a wonderful tool for teaching, and working in, computational linguistics using Python,” and “an amazing library to play with natural language.”
        Natural Language Processing with Python provides a practical introduction to programming for language processing. Written by the creators of NLTK, it guides the reader through the fundamentals of writing Python programs, working with corpora, categorizing text, analyzing linguistic structure, and more. The online version of the book has been been updated for Python 3 and NLTK 3. (The original Python 2 version is still available at http://nltk.org/book_1ed.)
        """
    B = """
        Jump Force is a celebration of 50 years of Weekly Shonen Jump manga, featuring nearly four dozen fighters from 16 of the magazine's most iconic stories. Bandai Namco's arena tag-team fighting game borrows plenty of elements from its source materials, for better and worse. Although Jump Force's campaign story drags on for way too long and ignores what could have been interesting character interactions in favor of repeated excuses for everyone to punch the crap out of each other, its combat is an enjoyable dance between two teams of fighters--thanks to the game's excellent mechanics and flashy visuals.
        In Jump Force, you're an ordinary human who's caught up in a warzone when the Dragon Ball, One Piece, and Naruto universes collide into our world and bring their assortment of heroes and villains with them. After being mortally wounded by Frieza, you're resurrected as a hero capable of learning the powers, skills, and abilities of Shonen Jump's characters, and you decide to join Goku, Luffy, and Naruto's Jump Force of allies in order to fix everyone's broken world. What follows is a fairly stereotypical shonen affair, with your character growing stronger over time, enemies and friends switching sides, and a mysterious evil working behind the scenes. Like most fighting games, there's not a single problem you don't ultimately just fix with your fists, from deciding team leader to knocking sense into those who have been corrupted by the same evil forces responsible for everyone's worlds colliding with one another.
        """