from sys import argv
from sys import exit
import loadAndSave as sl
import dataSelection as ds
from baseLinePredictions import OFMPredictions
from baseLinePredictions import GroupedPredictions
from wordLists import getWordList
from gensim.models import Word2Vec
from copy import deepcopy
from random import seed
from numpy import std
from numpy import mean
from configparser import SafeConfigParser
from configValidation import validateConfigFile
import time
import json
import collections
import os


def runGroupedTest(data, method, model, accuracyMeasure):
	"""
	Runs a grouped evaluation problem prediction on the given data and returns
	the accuracy using the selected accuracy measure.

	Args:
	data: The data to perform the prediction on.
	method: The selection of the prediction method to be used, valid arguments
	are 'random', 'wordCrossover' or 'word2vec'
	model: A trained word2vec model if method is 'word2vec' else None.
	accuracyMeasure: The measure by which the accuracy will be measured either
	'total' or 'pairs'.

	Returns:
	The accuracy as a float of using the selected prediction method on the
	given data using the selected accuracy measure.

	"""
	dataTest = GroupedPredictions()
	groupTestData = ds.createGroupedTestData(data)
	# sl.saveGroupedData('oxfordGroupedTest', groupTestData)
	if method == 'random':
		selections = dataTest.randomSelection(groupTestData, 3)
	elif method == 'wordCrossover':
		selections = dataTest.wordCrossoverSelection(groupTestData, 3)
	elif method == 'word2vec':
		selections = dataTest.word2VecSimilaritySelection(groupTestData, 3, model)

	if accuracyMeasure == 'total':
		return dataTest.calculateAccuracy(selections, groupTestData)
	elif accuracyMeasure == 'pairs':
		return dataTest.calculateAccuracyPairs(selections, groupTestData)


def runOFMTest(data, method, model):
	"""
	Runs a select one sentence from many options evaluation problem prediction
	on the given data and returns the accuracy.

	Args:
	data: The data to perform the prediction on.
	method: The selection of the prediction method to be used, valid arguments
	are 'random', 'wordCrossover', 'word2vecCosine' and 'word2vecWordSim'.
	model: A trained word2vec model if method is 'word2vecCosine' or
	'word2vecWordSim' else None.

	Returns:
	The accuracy as a float of using the selected prediction method on the
	given data.
	"""
	ofmPredictor = OFMPredictions()
	ofmData = ds.createOFMData(data)
	# sl.saveOneFromManyData('delete', ofmData)
	if method == 'random':
		selections = ofmPredictor.randomSelection(ofmData)
	elif method == 'wordCrossover':
		selections = ofmPredictor.wordCrossoverSelection(ofmData)
	elif method == 'word2vecCosine':
		selections = ofmPredictor.word2VecSimilaritySelectionCosine(ofmData, model)
	elif method == 'word2vecWordSim':
		selections = ofmPredictor.word2VecSimilaritySelectionWordSim(ofmData, model)
	return ofmPredictor.calculateAccuracy(selections, ofmData)


def main(argv):
	"""
	Runs evaluation of a prediction technique on a selected evaluation problem
	from a selected dataset. Runs the evaluation multiple times and prints stats
	to output. Takes as an argument the file path to a configeration file that
	is used to set the parameters of the evaluation.
	"""
	startTime = time.time()
	parser = SafeConfigParser()
	parser.read(argv[0])

	validConfig = validateConfigFile(parser)
	if not validConfig:
		print('[ERROR] - Config not valid!')
		exit()

	seed(parser.getint('evaluation_params', 'seedNo'))
	# print('Remove stop words: {} Remove punctuation: {} Lemmatize: {}'.format(rmStopwords, rmPunct, lemmatize))
	dictionaryDataPath = parser.get('evaluation_params', 'dictionary')
	try:
		evaluationData = sl.loadDataFromFile('dictionaryData/' + dictionaryDataPath)
	except IOError as err:
		print((dictionaryDataPath + ' can not be found in the dictionaryData directory.'))
		exit()

	evaluationData = ds.selectPoS(evaluationData, parser.get('evaluation_params', 'pos'))
	evaluationData = ds.removeWordsWithTooFewSenses(evaluationData,
													parser.getint('evaluation_params', 'numOfSenses'),
													parser.getint('evaluation_params', 'numOfExamp'))
	evaluationData = ds.examplesToLowerCase(evaluationData)
	evaluationData = ds.tokenizeAndLemmatizeExamples(evaluationData,
													 parser.getboolean('evaluation_params', 'lemmatize'))
	evaluationData = ds.removeStopwordsAndPunct(evaluationData,
												parser.getboolean('evaluation_params', 'rmStopwords'),
												parser.getboolean('evaluation_params', 'rmPunct'))

	num_examples_lt_5 = 0
	num_examples_lt_10 = 0
	num_examples_lt_20 = 0
	num_examples_all = 0
	base_path = '/Users/thomas/DevSandbox/EpicDataShelf/tag-lab/sense_alloc'
	pos = parser.get('evaluation_params', 'pos')
	with open(os.path.join(base_path, dictionaryDataPath.split('.')[0], '{}_lt_5.txt'.format(pos)), 'w', encoding='utf-8') as lt_5_file, \
			open(os.path.join(base_path, dictionaryDataPath.split('.')[0], '{}_lt_10.txt'.format(pos)), 'w', encoding='utf-8') as lt_10_file, \
			open(os.path.join(base_path, dictionaryDataPath.split('.')[0], '{}_lt_20.txt'.format(pos)), 'w', encoding='utf-8') as lt_20_file, \
			open(os.path.join(base_path, dictionaryDataPath.split('.')[0], '{}_all.txt'.format(pos)), 'w', encoding='utf-8') as all_file:
		for key, vals in list(evaluationData.items()):
			print(['### ITEM: {} ###'.format(key)])
			# print('\t{}\n'.format(json.dumps(vals, indent=4)))
			for val in vals:
				ex_lt_5 = []
				ex_lt_10 = []
				ex_lt_20 = []
				ex_all = []
				for ex in val['examples']:
					if (ex['sent'].count(' ') < 5):
						num_examples_lt_5 += 1
						ex_lt_5.append((ex['sent'], val['def']))
					if (ex['sent'].count(' ') < 10):
						num_examples_lt_10 += 1
						ex_lt_10.append((ex['sent'], val['def']))
					if (ex['sent'].count(' ') < 20):
						# print('{}: {}'.format(ex['sent'].count(' '), ex['sent']))
						num_examples_lt_20 += 1
						ex_lt_20.append((ex['sent'], val['def']))
					num_examples_all += 1
					ex_all.append((ex['sent'], val['def']))

				for l, f in [(ex_lt_5, lt_5_file), (ex_lt_10, lt_10_file), (ex_lt_20, lt_20_file), (ex_all, all_file)]:
					if (len(l) > 2):
						for ex, syn in l:
							f.write('{}\t{}\t{}\n'.format(key, ex, syn))

			print('---------------------------------------------------')
	print(('<=5: {}; <=10: {}; <=20: {}'.format(num_examples_lt_5, num_examples_lt_10, num_examples_lt_20)))

	"""
	model = None
	if 'word2vec' in parser.get('evaluation_params', 'baseLineMethod'):
		model = Word2Vec.load_word2vec_format(parser.get('evaluation_params',
			'word2vecBin'), binary=True)

	if len(evaluationData) < 1:
		print('Insufficient data to run evaluation. Try lowering the number ' +
			'of sense or examples required and try again.')
		exit()

	total = []
	for i in range(parser.getint('evaluation_params', 'testItterations')):
		dataSelected = ds.selectExamplesAndSenses(evaluationData,
			parser.getint('evaluation_params', 'numOfSenses'),
			parser.getint('evaluation_params', 'numOfExamp'))
		if parser.getboolean('evaluation_params', 'grouped'):
			total.append(runGroupedTest(dataSelected,
				parser.get('evaluation_params', 'baseLineMethod'), model,
				parser.get('evaluation_params', 'groupedAccuracyMeasure')))
		else:
			total.append(runOFMTest(dataSelected,
				parser.get('evaluation_params', 'baseLineMethod'), model))

	print('Average: {}'.format(mean(total)))
	print('Maximum: {}'.format(max(total)))
	print('Minimum: {}'.format(min(total)))
	print('Standard deviation: {}'.format(std(total)))
	print("{} seconds".format(time.time() - startTime))
	"""


if __name__ == '__main__':
	main(argv[1:])

