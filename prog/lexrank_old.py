import nltk.data
import numpy as np
import math
from sklearn.preprocessing import normalize
import random

# Convergence criterion for power iteration
EPSILON = 0.0001;

# Probability of surfer getting bored (for power iteration)
BORED = 0.15

MAX_POWER_ITERS = 1000

QUERY_WEIGHT = 1.5

LEXRANK_WEIGHT = 1.0

# produce list of sections
def text2seclist(filename, tkn):
    res = []
    with open(filename) as f:
        currlist = []
        counter = 0
        for row in f:
            if len(row) > 3 and row[:2] == '==' and not row[:3] == '===' and counter != 0:
                res.append(currlist)
                currlist = []
                
            counter += 1
            if len(row) > 0 and row[0] != '=':
				rowlist = tkn.tokenize(row.decode('utf-8').strip())
				rowlistli = [sent.split() for sent in rowlist]
				currlist.extend(rowlistli)
            
        res.append(currlist)
    return res

# produce list of sentences
def text2list(filename, tkn):
    res = []
    with open(filename) as f:
        currlist = []
        counter = 0
        for row in f:
            if len(row) > 0 and row[0] != '=':
				rowlist = tkn.tokenize(row.decode('utf-8').strip())
				rowlistli = [sent.split() for sent in rowlist]
				res.extend(rowlistli)
    return res


# load idfs
	
idfarr = np.loadtxt('enidf.txt', dtype=object)
idfs = {str(x[0]): float(x[1]) for x in idfarr}
max_idf = max(idfs.values())

def get_idf(word):
    if idfs.has_key(word):
        return idfs[word]
    else:
        return max_idf
				
#Compute the cosine similarity score between two sentences
def cosine_similarity(index_s1, index_s2, inverted_index, DocCluster):
    #split_index_s1 = index_s1.split("_")
    #section_s1 = int(split_index_s1[0])
    #sentence_s1 = int(split_index_s1[1])
    #Compute the length for the first sentence
    length_s1 = compute_length(index_s1, inverted_index, DocCluster)

    #split_index_s2 = index_s2.split("_")
    #section_s2 = int(split_index_s2[0])
    #sentence_s2 = int(split_index_s2[1])
    #Compute the length for the second sentence
    length_s2 = compute_length(index_s2, inverted_index, DocCluster)
	
    res = 0.0
	
	#Merge the two sentences to get the words
    both_sentences = []
    both_sentences.extend(DocCluster[index_s1])
    both_sentences.extend(DocCluster[index_s2])
    wordlist = np.unique(both_sentences)
    for word in wordlist:
        # Get the postings list for the word
    	postingslist = inverted_index[word]
        if index_s1 in postingslist.keys():
            if index_s2 in postingslist.keys():
                res += postingslist[index_s1] * postingslist[index_s2] * math.pow(get_idf(word), 2)
                
    res = res/(length_s1*length_s2)
    return res
    
def cosine_similarity_title(title, index_s, inverted_index, DocCluster, lower=False):
    #split_index_s1 = index_s1.split("_")
    #section_s1 = int(split_index_s1[0])
    #sentence_s1 = int(split_index_s1[1])
    #Compute the length for the first sentence
    #length_s1 = compute_length(index_s1, inverted_index, DocCluster)

    #split_index_s2 = index_s2.split("_")
    #section_s2 = int(split_index_s2[0])
    #sentence_s2 = int(split_index_s2[1])
    #Compute the length for the second sentence
    length_s = compute_length(index_s, inverted_index, DocCluster)
	
    res = 0.0
	
	#Merge the two sentences to get the words
    both_sentences = []
    both_sentences.extend(title)
    #both_sentences.extend(DocCluster[index_s])
    wordlist = np.unique(both_sentences)
    for word in wordlist:
        # Get the postings list for the word
        if word in inverted_index.keys():
            postingslist = inverted_index[word]
            if index_s in postingslist.keys():
                #if index_s in postingslist.keys():
                res += postingslist[index_s] * (get_idf(word.lower()) if lower else get_idf(word)) 
        else:
            res += 0.0    
               
    res = res/(length_s)
    return res
                
# Compute the length of the tf-idf vector for a given sentence
def compute_length(index_s, inverted_index, DocCluster):
    res = 0.0
    wordlist = np.unique(DocCluster[index_s])
    for word in wordlist:
        postingslist = inverted_index[word]
        res += math.pow(postingslist[index_s]*get_idf(word), 2)
    res = math.sqrt(res)
    return res
    
def construct_tf_index(doccluster):
	# Construct tf index
	tfs = {}
	for index_s in range(len(doccluster)):
	    sentence = doccluster[index_s]
	    for word in sentence:
	        if word not in tfs.keys():
	            tfs[word] = {}
	        if index_s not in tfs[word].keys():
	            tfs[word][index_s] = 1
	        else:
	            tfs[word][index_s] += 1
	            
	return tfs

def normdiff(a, b):
    try:
	    return sum([abs(a[i] - b[i]) for i in range(len(a))])
    except RuntimeWarning:
        print a, '\n', b

def getCoefMatrix(i, j, nbOfDocs, graf, ut):
    #sink
    if ut[i] == 0:
    	if i == j:
        	return BORED/nbOfDocs
      	else:
            return (1.0-BORED)/(nbOfDocs - 1) + BORED/nbOfDocs
	
    # not a sink
    else:
        if j in graf[i].keys():
            return (1.0 - BORED) / ut[i] + BORED / nbOfDocs
        else:
            return BORED / nbOfDocs

# Step for power iteration
def step(v, graf, ut):
    next = np.zeros(len(v), dtype=np.float64)
    for j in range(len(next)):
        for i in range(len(next)):
            next[j] += getCoefMatrix(i, j, len(next), graf, ut) * v[i]
    return next
    
def step2(x, ninv, graph, out):
    G = np.zeros([len(x),len(x)],dtype=np.float64)
    res = np.zeros(len(x))
    for i in range(len(x)):
        if out[i] == 0:
            for j in range(len(x)):
                if i == j:
                    res[j] += x[i] * BORED * ninv
                    G[i][j] = BORED * ninv
                else:
                    res[j] += x[i] * (1.0 - BORED) / (len(x) - 1) + BORED * ninv
                    G[i][j] = (1.0 - BORED) / (len(x) - 1) + BORED * ninv
        else:
            outprob = (1.0 - BORED) * (1.0 / out[i]) + BORED * ninv
            nooutprob = BORED * ninv
            for j in range(len(x)):
                if j not in graph[i].keys():
                    res[j] += x[i] * nooutprob
                    G[i][j] = nooutprob
                else:
                    res[j] += x[i] * outprob
                    G[i][j] = outprob
    print G
    return res
    
def poweriteration(nsentences, graph, out):
    x = np.zeros(nsentences, dtype=np.float64)
    xt = np.zeros(nsentences, dtype=np.float64)
    xt[0] = 1.0
	
    ninv = 1.0 / nsentences
    iters = 0
    while normdiff(x,xt) > EPSILON and iters < MAX_POWER_ITERS:
        print "normdiff", normdiff(x, xt)
        x = xt[:]
        xt = step(x, graph, out)
        #print xt
        #xt = step2(x, ninv, graph, out)
        #xt = normalize(xt[:,np.newaxis], axis=0).ravel()
        #print x[:5]
        #print xt[:5]
        iters += 1

    return xt

def followrandomlink(graph, out, node, nsents):
    if out[node] == 0:
        return random.randint(0, nsents-1)
    else:
        return random.choice(graph[node].keys())

def randompath(graph, out, nwalks, nsents, randomstart, countvisits, breakdangling):		
    outerstop = 1
    innerstop = nwalks
    if not randomstart:
        outerstop = nwalks / nsents
        innerstop = nsents
	
	nvisits = 0
	timesvisited = {}
	for i in range(outerstop):
	    for j in range(innerstop):
	        bored = False
	        currnode = random.randint(0, nsents-1) if randomstart else j
	        while not bored:
	            if countvisits:
	                if currnode not in timesvisited.keys():
	                    timesvisited[currnode] = 1.0
	                else:
	                    timesvisited[currnode] += 1.0
	                nvisits += 1
	          
	            if breakdangling and out[currnode] == 0:
	                break
	             
	            if random.random() < BORED:
	                bored = True
	                if not countvisits:
	                    if currnode not in timesvisited.keys():
	                        timesvisited[currnode] = 1.0
	                    else:
	                        timesvisited[currnode] += 1.0
	            else:
	                currnode = followrandomlink(graph, out, currnode, nsents)
      
    # normalize
    for node in timesvisited.keys():
        denom = nvisits if countvisits else (nwalks if randomStart else outerstop * innerstop)
        timesvisited[node] = timesvisited[node] / denom
    return timesvisited
		
def main():	
    # load texts
    sd = nltk.data.load('tokenizers/punkt/english.pickle')
    doc1list = text2list('docs/National_Basketball_Association', sd)
    title = ['National','Basketball','Association']
    title_lower = [x.lower() for x in title]
    #doc2list = text2list("docs/Women's_National_Basketball_Association", sd)
    condoc = doc1list#np.hstack((doc1list, doc2list))
    nsentences = len(condoc)

    # load idfs
    idfarr = np.loadtxt('enidf.txt', dtype=object)
    idfs = {str(x[0]): float(x[1]) for x in idfarr}
    max_idf = max(idfs.values())
    print 'max idf', max_idf

    tfs = construct_tf_index(condoc)
	
    threshold = 0.5 # test (change)
	
    # pairwise cosine between all sentences.
    # edge if above threshold, undirected
    graph = {}
    out = np.zeros(nsentences)
        
    # compute similarities with title
    title_sims = np.zeros(nsentences, dtype=float)
    for index_s in range(nsentences):
        cosim =  cosine_similarity_title(title, index_s, tfs, condoc)
        title_sims[index_s] = cosim
    
    title_sims_lower = np.zeros(nsentences, dtype=float)
    for index_s in range(nsentences):
        cosim =  cosine_similarity_title(title_lower, index_s, tfs, condoc)
        title_sims_lower[index_s] = cosim 
   
    title_sims = [x**2 for x in title_sims]
    title_sims = title_sims/np.sqrt(sum(title_sims))
    #title_sims = normalize(title_sims, axis=0)
        
    nedges = 0
    for index_s1 in range(nsentences):
        for index_s2 in range(index_s1 + 1, nsentences):
            cosim = cosine_similarity(index_s1, index_s2, tfs, condoc)
            if cosim > threshold:
                nedges += 1
                if index_s1 not in graph.keys():
                    graph[index_s1] = {}
                if index_s2 not in graph.keys():
                    graph[index_s2] = {}
                graph[index_s1][index_s2] = 1
                graph[index_s2][index_s1] = 1
                out[index_s1] += 1
                out[index_s2] += 1       
        
       
    print "Edges in graph", nedges
    #print out
    
    eigenv = poweriteration(nsentences, graph, out)
    eigenv = [LEXRANK_WEIGHT * eigenv[i] + QUERY_WEIGHT * 0.85 * title_sims[i] \
        + QUERY_WEIGHT * 0.15 * title_sims_lower[i] for i in range(len(title_sims))]
    #eigenv = randompath(graph, out, nsentences * 10, nsentences, False, True, False) 
    #print eigenv
    sentencemap = zip(range(nsentences), eigenv)
    #sentencemap = [(x,eigenv[x]) for x in eigenv.keys()] 
    #print sentencemap
    
    #sentencemap = sorted(sentencemap, key=lambda x: x[1], reverse=True)
    sentencemap = sorted(sentencemap, key=lambda x: x[1], reverse=True)
    #print sentencemap
    #bestsentences = [x[0] for x in sentencemap[:10]]
	
    for index_s, score in sentencemap[:10]:
        print '(',index_s,',',score,')',' '.join(condoc[index_s])

    #print graph  
    print nsentences  
    
def testPowerIter():
    graph={}
    graph[0] = {}
    graph[0][1] = 1
    graph[0][2] = 1
    graph[0][3] = 1
    graph[1] = {}
    graph[1][3] = 1
    graph[2] = {}
    graph[2][3] = 1
    graph[2][4] = 1
    graph[3] = {}
    graph[3][4] = 1
    out = [3,1,2,1,0]
    nsentences = 5
    poweriteration(nsentences, graph, out)
    
main()
#testPowerIter()
