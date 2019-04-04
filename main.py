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


# unigram tagger
unigram_tagger = nltk.UnigramTagger(train_sents)
print('Unigram tagger eval: ' + str(unigram_tagger.evaluate(test_sents)))

output = open('unigramTagger.pkl', 'wb')
dump(unigram_tagger, output, -1)
output.close()
print('Unigram tagger output saved')


# tnt tagger
tnt_tagger = nltk.TnT()
tnt_tagger.train(train_sents)
print('TnT tagger eval: ' + str(tnt_tagger.evaluate(test_sents)))

output = open('tntTagger.pkl', 'wb')
dump(unigram_tagger, output, -1)
output.close()
print('TnT tagger output saved')


# perceptron tagger
perceptron_tagger = nltk.PerceptronTagger(load=False)
perceptron_tagger.train(train_sents)
print('Perceptron tagger eval: ' + str(perceptron_tagger.evaluate(test_sents)))

output = open('perceptronTagger.pkl', 'wb')
dump(unigram_tagger, output, -1)
output.close()
print('Perceptron tagger output saved')


# crf tagger
crf_tagger = nltk.tag.CRFTagger()
crf_tagger.train(train_sents)
print('CRF tagger eval: ' + str(crf_tagger.evaluate(test_sents)))

output = open('crfTagger.pkl', 'wb')
dump(unigram_tagger, output, -1)
output.close()
print('CRF tagger output saved')
