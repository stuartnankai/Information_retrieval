####ROUGE-1####
import re



#returns the ROUGE-1 score of the summary in path_summary with respect to
#the human summary in path_human_summary
def evaluate(path_summary, path_human_summary):
    matches = countMatches(listOfWords(path_summary), listOfWords(path_human_summary));  
    return float(matches)/float(length(path_human_summary));

#counts the 1-gram matches 
def countMatches(summary, human_summary):
    count = 0;
    for word in human_summary:
	if word in summary:
		count += 1;
    return count;

#returns the length of a summary located at path
def length(path):
    return len(listOfWords(path));
	
#returns a list of all words in lower cases from a path
def listOfWords(path):
    text = ''.join(open(path).readlines())
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
    all_words =  []
    for sentence in sentences:
	tmp_sent = sentence.split()
	for word in tmp_sent:
		word = word.lower()
		all_words.append(word.strip(','))
    return all_words


print("The ROUGE-1 score is: ", evaluate('test_summary.txt', 'test_human_summary.txt'));
print("The ROUGE-1 score is: ", evaluate('test_summary2.txt', 'test_human_summary.txt'));
