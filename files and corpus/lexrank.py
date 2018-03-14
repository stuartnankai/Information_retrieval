""" 
Automatic query dependant text summarization of one or several Wikipedia 
articles using the LexRank algorithm combined with some heuristic improvements.

Uses a list of idf scores by http://www.summarization.com/mead/

Examples of usage from another Python script:

    res = summarize(['Toilet','paper'])
    res = summarize(['Toilet','paper'], nres=5, nart=3, montecarlo=3, printouts=True, verbose=False)


Examples of usage from command line:
    ./lexrank.py Toilet paper
    ./lexrank.py Toilet paper -nres 5 -nart 3 -v -mc 3 -he -lv 1
    
    
# TODO: Remove duplicate sentences if any
""" 

import nltk.data
import numpy as np
import random
import wikipedia
import sys
import argparse
from improveLexRankJian import improveLexRank as jianRank

# Convergence criterion for power iteration
EPSILON = 0.0001;

# Probability of surfer getting bored (for power iteration)
BORED = 0.15

# max number of iterations in the power iteration algorithm
MAX_POWER_ITERS = 1000

# Weight of the query similarity score in the final score
#QUERY_WEIGHT = 1.0#1.5

# Weight of the LexRank score in the final score.
#LEXRANK_WEIGHT = 1.0#1.0#1.0

# An edge in the LexRank graph is created when the cosine similarity between 
# sentences is greater than the threshold
LEXRANK_THRESHOLD_MIN = 0.1#0.5

# Relevance decreases by the following factor for each sentence
RELEVANCE_FACTOR_SENTENCE = 0.8

# Relevance decreases by the following factor for each section
RELEVANCE_FACTOR_SECTION = 0.8

RELEVANCE_FACTOR_ARTICLE = 0.8

STOP_WORDS = ['==References==', '== References ==',
              '==See also==','== See also ==',
              '==See Also==','== See Also ==',
              '==Further reading==', '== Further reading ==',
              '==Further Reading==', '== Further Reading ==']
       
      
# load idfs
idfarr = np.loadtxt('enidf.txt', delimiter=None, dtype=(str,float), usecols=(0,1))
with open('enidf.txt') as idff:
    idfarr = [x.strip('\n').split() for x in idff.readlines()]

idfs = {str(x[0]): float(x[1]) for x in idfarr}
max_idf = max(idfs.values())

def get_idf(word):
    if word in idfs.keys(): #has_key(word):
        return idfs[word]
    else:
        return max_idf

# produce list of sentences wiki page
def wiki2list(page, tkn, start_index=0, tokenize_sentences=True):
    sentlist = []
    sectionindices = [start_index]
    paragraphindices = []
    counter = start_index
    lines = page.splitlines()
    for row in lines:
        if len(row) > 0 and row[0] != '=':
            rowlist = tkn.tokenize(row.strip())
            if tokenize_sentences:
                rowlistli = [sent.strip('.').strip(';').split() for sent in rowlist] #strip more?
                sentlist.extend(rowlistli)
            else:
                sentlist.extend(rowlist)
            paragraphindices.append(counter)
            counter += len(rowlist)
        
        if len(row) > 2 and row[:2] == '==' and row[:3] != '===':
            if row in STOP_WORDS:
               break
            else:
                sectionindices.append(counter)
	        
    return sentlist, sectionindices, paragraphindices
				
#Compute the cosine similarity score between two sentences
def cosine_similarity(index_s1, index_s2, inverted_index, DocCluster):
    length_s1 = compute_length(index_s1, inverted_index, DocCluster)
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
                res += postingslist[index_s1] * postingslist[index_s2] * (get_idf(word)**2)
                
    if res > 0.0:
        res = res/(length_s1*length_s2)
        return res
    return 0.0

# Compute the cosine similarity with the query for all sentences
def cosine_similarity_query(query, inverted_index, DocCluster, nsentences):
    query_sims = np.zeros(nsentences, dtype=float)
    
    both_sentences = []
    both_sentences.extend(query)
    wordlist = np.unique(both_sentences)
    for word in wordlist:
        if word in inverted_index.keys():
            postingslist = inverted_index[word]
            for index_s in postingslist.keys():
                query_sims[index_s] += postingslist[index_s] * get_idf(word)

    for index_s in range(nsentences):
        if query_sims[index_s] > 0.0:
            length_s = compute_length(index_s, inverted_index, DocCluster)
            query_sims[index_s] = query_sims[index_s]/length_s

    return query_sims
                
# Compute the length of the tf-idf vector for a given sentence
def compute_length(index_s, inverted_index, DocCluster):
    res = 0.0
    wordlist = np.unique(DocCluster[index_s])
    for word in wordlist:
        postingslist = inverted_index[word]
        res += (postingslist[index_s]*get_idf(word))**2
    res = res**.5
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
        print(a, '\n', b)

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
    
def poweriteration(nsentences, graph, out, verbose=False):
    x = np.zeros(nsentences, dtype=np.float64)
    xt = np.zeros(nsentences, dtype=np.float64)
    xt[0] = 1.0
	
    ninv = 1.0 / nsentences
    iters = 0
    while normdiff(x,xt) > EPSILON and iters < MAX_POWER_ITERS:
        if verbose:
            print("normdiff", normdiff(x, xt))
        x = xt[:]
        xt = step(x, graph, out)
        iters += 1

    return xt

def followrandomlink(graph, out, node, nsents):
    if out[node] == 0:
        return random.randint(0, nsents-1)
    else:
        return random.choice(list(graph[node].keys()))
        
#false,true,false
def randompath(graph, out, nwalks, nsents, randomstart, countvisits, breakdangling):		
    outerstop = 1
    innerstop = nwalks
    if not randomstart:
        outerstop = int(nwalks / nsents)
        innerstop = int(nsents)
	
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
	
	
def searchwikipedia(querystr, narticles, lexrankversion):
    # Search for articles
    search_results = wikipedia.search(querystr, narticles)
    
    # Load sentence tokenizer
    tknz = nltk.data.load('sentence_tokenizer.pickle')
    
    # Retrieve articles into a list of sentences while noting section indices
    doc_collection = [] # Collection of all documents parsed so far.
    article_start_index = 0
    article_indices = []
    section_indices = []
    paragraph_indices =[]
    doclist = [] # Temporary variable (List of sentences in current document)
    
    error_msgs = []
    article_counter = 0
    for title in search_results:
        
        try:
            article = wikipedia.page(title).content
        except wikipedia.exceptions.DisambiguationError as e:
        
            # Very likely that we would give a bad summary if we miss
            # all the pages in this particular disambiguation list
            if title == querystr or title.lower() == querystr.lower() \
                or article_counter == 0:
                
                return False, [e.options]
                
            error_msgs.append(e.options)
            article_counter += 1
            continue
              
        article_start_index += len(doclist)
        article_indices.append(article_start_index)
        doclist, a_section_indices, a_paragraph_indices = wiki2list(article, tknz, 
            article_start_index, lexrankversion == 1)
        section_indices.extend(a_section_indices)
        paragraph_indices.extend(a_paragraph_indices)
        doc_collection.extend(doclist)
        article_counter += 1
   
    if len(error_msgs) == len(search_results):
        if verbose:
            print(error_msgs)
        return False, error_msgs
    nsentences = len(doc_collection)

    # Remove identical sentences
    toRemove = []
    for index_s1 in range(nsentences):
        for index_s2 in range(index_s1 + 1, nsentences):
            if doc_collection[index_s1] == doc_collection[index_s2]:
                toRemove.append(index_s2)

    for index in range(nsentences):
        if index in toRemove:
            doc_collection.pop(index)
            article_indices = [x if x <= index else x-1 for x in article_indices]
            section_indices = [x if x <= index else x-1 for x in section_indices]
            paragraph_indices = [x if x <= index else x-1 for x in paragraph_indices]
   
    return doc_collection, article_indices, section_indices, paragraph_indices
	
# LexRank 1.0 / CharlesDanielRank
def cdrank(doc_collection, tfs, montecarlo, verbose):
    # pairwise cosine between all sentences.
    # edge if above threshold, undirected
    graph = {}
    nsentences = len(doc_collection)
    out = np.zeros(nsentences)
    nedges = 0
    # List of sentences that are exactly the same
    sameSentences = []
    for index_s1 in range(nsentences):
        for index_s2 in range(index_s1 + 1, nsentences):
            cosim = cosine_similarity(index_s1, index_s2, tfs, doc_collection)
            if cosim > LEXRANK_THRESHOLD_MIN and len(doc_collection[index_s1]) > 9 \
                and len(doc_collection[index_s2]) > 9:
                nedges += 1
                if index_s1 not in graph.keys():
                    graph[index_s1] = {}
                if index_s2 not in graph.keys():
                    graph[index_s2] = {}
                graph[index_s1][index_s2] = 1
                graph[index_s2][index_s1] = 1
                out[index_s1] += 1
                out[index_s2] += 1       
        
    if verbose:  
        print("Edges in graph", nedges)
        #print "cosim ex: ", cosim = cosine_similarity(index_s1, index_s2, tfs, doc_collection)
  
    # Monte Carlo  
    if montecarlo:
        eigendict = randompath(graph, out, nsentences * montecarlo, nsentences, 
            False, True, False)
        eigenv = [eigendict[i] for i in range(len(eigendict))]
    
    # Power iteration
    else:
        eigenv = poweriteration(nsentences, graph, out, True if verbose else False)
            
    return eigenv
     
     
def apply_heuristics(query_list, doc_collection, tfs, artindices, secindices, eigenv, 
    queryw=1.0, lexrankw=1.0, reldecreasefactor=0.8):
    
    nsentences = len(doc_collection)
    
    # compute similarities with query
    query_sims = cosine_similarity_query(query_list, tfs, doc_collection, nsentences)
    query_sims_norm = [x**2 for x in query_sims]
    query_sims = query_sims/np.sqrt(sum(query_sims_norm))
    
    # Decrease relevance
    scores = np.zeros(len(eigenv))
    section_coef = 1.
    sentence_coef = 1.
    article_coef = 1.
    nextsection_ii = 1 # index in section index (ii)
    nextarticle_ii = 1
    for i in range(len(eigenv)):
        #new article
        if nextarticle_ii < len(artindices) and i >= artindices[nextarticle_ii]:
            nextarticle_ii += 1
            section_coef = 1.
            sentence_coef = 1.
            article_coef *= reldecreasefactor
        
        # new section
        if nextsection_ii < len(secindices) and i >= secindices[nextsection_ii]:
            nextsection_ii += 1 
            section_coef *= reldecreasefactor
            sentence_coef = 1.
        
        # update
        scores[i] = (lexrankw * eigenv[i] + queryw * query_sims[i]) * \
                     section_coef * sentence_coef * article_coef       
        sentence_coef *= reldecreasefactor
    
    return scores

def process_query(query_list):
    query_str = " ".join(query_list)
    query_list.extend([query_list[0].lower()])
    query_list[0] = query_list[0][0].upper() + query_list[0][1:]
    return query_list, query_str    
    
#only for lexrank 1 for now
def summarize_list(query_list, doc_collection, tfs, eigenv, nres, artindices, 
    secindices, queryw, lexrankw, reldecreasefactor):
    
    scores = apply_heuristics(query_list, doc_collection, tfs, artindices, secindices, 
        eigenv, queryw, lexrankw, reldecreasefactor)
    
    sentencemap = zip(range(len(doc_collection)), scores)
    sentencemap = sorted(sentencemap, key=lambda x: x[1], reverse=True)
    res = [' '.join(doc_collection[index_s]) for index_s,_ in sentencemap[:nres]]
    return res
      
 
def summarize(query, nres=10, nart=2, lexrankversion=1, queryw=1.0, lexrankw=1.0, 
    reldecreasefactor=0.8, montecarlo=None, heuristics=False, printouts=False, verbose=False):

    """ Query dependent summarization of one or several Wikipedia articles 
    
    Parameters: 
        lexrankversion: lexrank version: 1 Charles-Daniel, 2 Jian
        montecarlo:     Number of start from each node in the graph. E.g. if 3 
                        is given, 3*N starts are made
        heuristics:     use heuristics
        
    Returns:
        boolean:        whether a summary could be produced or not.
        list:           summary or information about error. 
    """
    
    if verbose:
        print('summarize(',query,', ',nres,', ',nart,', ',lexrankversion, ',', montecarlo, ',..)')

    query, querystr = process_query(query)
    doc_collection, artindices, secindices, parindices = searchwikipedia(querystr, 
        nart, lexrankversion)

    nsentences = len(doc_collection)
    
    if verbose:
        print("section indices", secindices)
        print("article indices", artindices)
        print('max idf', max_idf)
        print('#sentences', nsentences)
    
    eigenv=[]
    if lexrankversion == 1:
        ### CHARLES & DANIEL LEXRANK ###
        tfs = construct_tf_index(doc_collection)
        eigenv = cdrank(doc_collection, tfs, montecarlo, verbose)
    
    else:
        ### JIAN LEXRANK ###
        eigenv = jianRank(doc_collection, verbose)
    
    if lexrankversion == 2:
        full_doc_collection = doc_collection[:]
        doc_collection = [x.strip('.').strip(';').split() for x in doc_collection]
        tfs = construct_tf_index(doc_collection)
    
    if heuristics:
        apply_heuristics(query, doc_collection, tfs, artindices, secindices, eigenv, 
            queryw=1.0, lexrankw=1.0, reldecreasefactor=0.8)
  
    else:
        scores = eigenv
        
    sentencemap = zip(range(nsentences), scores)
    sentencemap = sorted(sentencemap, key=lambda x: x[1], reverse=True)
    
    if lexrankversion == 1:
        res = [' '.join(doc_collection[index_s]) for index_s,_ in sentencemap[:nres]]
    else:
        res = [full_doc_collection[index_s] for index_s,_ in sentencemap[:nres]]
        
    if verbose:
        for index_s, score in sentencemap[:nres]:
            if lexrankversion == 1:
                print('(' + str(index_s), ',', str(score) + ')', ' '.join(doc_collection[index_s]))
            else:
                print('(' + str(index_s), ',', str(score) + ')', full_doc_collection[index_s])
    elif printouts:
        if lexrankversion == 1:
            print('. '.join(res) + '.')
        else:
            print(' '.join(res))
            
    return True, res          
        
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
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Wikipedia text summarization 
    using a modified LexRank algorithm and other heuristic improvements. 
    Returns the summarized article as a string.''')
    parser.add_argument("query", nargs='+', help="Search query")
    parser.add_argument("-nres", type=int, default=10, help="number of returned sentences")
    parser.add_argument("-nart", type=int, default=2, help="number of Wikipedia articles to consider")     
    #parser.add_argument("-pa","--paragraph_decrease",help='''Decrease relevance 
    #with each new paragraph. Default is section-wise decrease''',
    #    action="store_true")
    parser.add_argument("-lv","--lexrankversion", type=int, default=1,help="""(1|2) use Charles-Daniel
    or Jian's lexrank""")
    parser.add_argument("-he", "--heuristics", help="use heuristics",
        action="store_true")
    parser.add_argument("-mc","--montecarlo", default=None, help='''number of 
    montecarlo approximation iterations. Default is 0 (power iteration instead).''')  
    parser.add_argument("-v","--verbose",help="verbose printouts",
        action="store_true")     
    args = parser.parse_args()

    success, res = summarize(printouts=True,**vars(args))
    if not success:
        print("Problem disambiguating the query. Try adding more information.")
              
    #testPowerIter()
