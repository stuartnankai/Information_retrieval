####ROUGE-1####
import re


uniGram = dict();
biGram = dict();
triGram = dict();
fourGram = dict();
#returns the ROUGE-N score of the summary in path_summary with respect to
#the human summary in path_human_summary
def evaluate(path_summary, path_human_summary):
	text = ''.join(open(path_summary).readlines()).replace(",", "")
	sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
	hashWords(sentences);
	human = ''.join(open(path_human_summary).readlines()).replace(",", "")
	human_summary = re.split(r' *[\.\?!][\'"\)\]]* *', human);
	print "Rouge-1 score: ", countUnigram(human_summary);
	print "Rouge-2 score: ", countBigram(human_summary);
	print "Rouge-3	score: ", countTrigram(human_summary);
	print "Rouge-4 score: ", countFourgram(human_summary);


def countUnigram(human_summary):
	count = float(0);
	total = float(0)
	for sentence in human_summary:
		sentence = sentence.split()
		for word in sentence:
			word = word.lower()
			total += 1;
			if (word in uniGram):
				count += 1;
	return (count)/(total)
	
def countBigram(human_summary):
	count = 0;
	total = 0;
	print biGram
	for sentence in human_summary:
		sentence = sentence.split()
		for i in range(0, len(sentence)-1):
			bigram = sentence[i].lower()+sentence[i+1].lower();
			total += 1;
			print bigram
			if (bigram in biGram):
				count += 1;
	return float(count)/float(total)
	
def countTrigram(human_summary):
	count = 0;
	total = 0;
	for sentence in human_summary:
		sentence = sentence.split()
		for i in range(0, len(sentence)-2):
			total += 1;
			trigram = sentence[i].lower()+sentence[i+1].lower()+sentence[i+2].lower();
			if (trigram in triGram):
				count += 1;
	return float(count)/float(total)
	
def countFourgram(human_summary):
	count = 0;
	total = 0;
	for sentence in human_summary:
		sentence = sentence.split()
		for i in range(0, len(sentence)-3):
			total += 1;
			fourgram = sentence[i].lower()+sentence[i+1].lower()+sentence[i+2].lower()+sentence[i+3].lower();
			if (fourgram in fourGram):
				count += 1;
	return float(count)/float(total)
#returns the length of a summary located at path
def length(path):
	return len(listOfWords(path));

def hashUnigram(sentences):
	for sentence in sentences:
		sentence = sentence.split()
		for word in sentence:
			word = word.lower()
			uniGram[word] = True;
def hashBigram(sentences):
	for sentence in sentences:
		sentence = sentence.split()
		for i in range(0, len(sentence)-1):
			bigram = sentence[i].lower()+sentence[i+1].lower()
			biGram[bigram] = True;
def hashTrigram(sentences):
	for sentence in sentences:
		sentence = sentence.split()
		for i in range(0, len(sentence)-2):
			trigram = sentence[i].lower()+sentence[i+1].lower()+sentence[i+2].lower()
			triGram[trigram] = True;
def hashFourgram(sentences):
	for sentence in sentences:
		sentence = sentence.split()
		for i in range(0, len(sentence)-3):
			fourgram = sentence[i].lower()+sentence[i+1].lower()+sentence[i+2].lower()+sentence[i+3].lower();
			fourGram[fourgram] = True;
#hash all N-grams for the summary found in path
def hashWords(sentences):
	hashUnigram(sentences);
	hashBigram(sentences);
	hashTrigram(sentences);
	hashFourgram(sentences);
	
	

print "SCORES: "
#evaluate('test_summary.txt', 'test_human_summary.txt');
print "SCORES: "
evaluate('test_summary2.txt', 'test_human_summary.txt');
