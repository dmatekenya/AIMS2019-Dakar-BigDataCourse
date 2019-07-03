def translate_to_french_for_dunstan(sentence=None):
    """
    Given a sentence, translate each word in the sentence
    Example: sentence = 'I love you', returns {"I": "je", "love": "amour", "you": "vous"}
    use textblob package (https://textblob.readthedocs.io/en/dev/) and NLTK package
    for this task
    :param sentence: Sentence to translate
    :return: a dictionary where key is english word and value is translated french word
    """
    # first tokenize the words: split the sentence
    # into words using the NLTK function word_tokenize()
    words = nltk.word_tokenize(sentence)

    # initiate a dictionary object to put in english and French words
    en_fr = {}

for w in words:
    en_blob = TextBlob(w)

    # use the function translate(from_lang="en", to='fr')
    # on the en_blob object defined above
    fr_blob = en_blob.translate(from_lang="en", to='fr')

    # use function raw on the blob above to get the word as a string
    fr_word = fr_blob.raw

    # put the translated word in the
    # dictionary object en_fr with english
    # as key and corresponding french translation as value
    en_fr[w] = fr_word

# return the dictionary object
return en_fr