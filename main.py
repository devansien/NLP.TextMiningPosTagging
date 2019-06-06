import nltk
from nltk.corpus import brown
from pickle import dump

# load brown
brown_tagged_sents = brown.tagged_sents(categories='news')
size = int(len(brown_tagged_sents) * 0.8)

# load train & test sets
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

# print sizes of both test & train sets
print(len(train_sents))
print(len(test_sents))

from nltk.corpus import brown
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
data = nltk.ConditionalFreqDist((word.lower(), tag) for (word, tag) in brown_news_tagged)

for word in sorted(data.conditions()):
     if len(data[word]) > 2:
        tags = [tag for (tag, _) in data[word].most_common()]
        print(word, ' '.join(tags))

# # unigram tagger
# unigram_tagger = nltk.UnigramTagger(train_sents)
# print(unigram_tagger[0])
# print('Unigram tagger eval: ' + str(unigram_tagger.evaluate(test_sents)))
#
# output = open('unigramTagger.pkl', 'wb')
# dump(unigram_tagger, output, -1)
# output.close()
# print('Unigram tagger output saved')
#
# # tnt tagger
# tnt_tagger = nltk.TnT()
# tnt_tagger.train(train_sents)
# print('TnT tagger eval: ' + str(tnt_tagger.evaluate(test_sents)))
#
# output = open('tntTagger.pkl', 'wb')
# dump(tnt_tagger, output, -1)
# output.close()
# print('TnT tagger output saved')
#
# # perceptron tagger
# perceptron_tagger = nltk.PerceptronTagger(load=False)
# perceptron_tagger.train(train_sents)
# print('Perceptron tagger eval: ' + str(perceptron_tagger.evaluate(test_sents)))
#
# output = open('perceptronTagger.pkl', 'wb')
# dump(perceptron_tagger, output, -1)
# output.close()
# print('Perceptron tagger output saved')
#
# # crf tagger
# crf_tagger = nltk.CRFTagger()
# crf_tagger.train(train_sents, 'crfTagger.pkl')
# print('CRF tagger eval: ' + str(crf_tagger.evaluate(test_sents)))
# print('CRF tagger output saved')
