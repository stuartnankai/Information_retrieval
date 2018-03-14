from rouge import evaluate
from lexrank import summarize, process_query, searchwikipedia, construct_tf_index, cdrank, summarize_list
import numpy as np
import nltk.data
import string, os, re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
#QUERY = ['Automatic','summarization']
#HUMAN_SUMMARY_FILE = 'human_summaries/Automatic_summarization.txt'
QUERY = ['Chick-fil-A']
HUMAN_SUMMARY_FILE = 'human_summaries/Chick-fil-A.txt'
#QUERY = ['Battle', 'of', 'Arawe']
#HUMAN_SUMMARY_FILE = 'human_summaries/Battle_of_Arawe.txt'

NARTICLES = 2

data_filename = 'rouge_data'+'_'.join(QUERY)+'.npz'
if os.path.exists(data_filename):
    rouge_data = np.load(data_filename)['rouge_data']
else:
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    with open(HUMAN_SUMMARY_FILE) as f:
        tknz = nltk.data.load('sentence_tokenizer.pickle')
        sentences = [tknz.tokenize(x.strip('\n')) for x in f.readlines()][0]
        human_summary = [regex.sub('', s).split() for s in sentences]
    
    # Perform LexRank
    query_list, query_str = process_query(QUERY)
    doc_collection, artindices, secindices, parindices = searchwikipedia(query_str, NARTICLES, 1)
    tfs = construct_tf_index(doc_collection)
    eigenv = cdrank(doc_collection, tfs, False, False)
    
    query_weights = np.arange(0.0, 1.01, 0.01)
    positional_factors = np.arange(0.01, 1.01, 0.01)
    
    #zs = np.array([score_summary(qw,pf) for qw,pf in zip(np.ravel(QW), np.ravel(PF))])
    #Z = zs.reshape(QW.shape)
    
    qwlen = len(query_weights)
    pflen = len(positional_factors)
    rouge1_scores = np.empty([qwlen, pflen])
    rouge2_scores = np.empty([qwlen, pflen])
    rouge3_scores = np.empty([qwlen, pflen])
    rouge4_scores = np.empty([qwlen, pflen])
    for i in range(len(query_weights)):
        for j in range(len(positional_factors)):
            summary_list = summarize_list(query_list, doc_collection, tfs, eigenv, 5, 
                artindices, secindices, query_weights[i], 1.0, positional_factors[j])
            summary = [regex.sub('', s).split() for s in summary_list]
            res = evaluate(human_summary, summary)
            rouge1_scores[i][j] = res[0]
            rouge2_scores[i][j] = res[1]
            rouge3_scores[i][j] = res[2]
            rouge4_scores[i][j] = res[3]
       
    rouge_data = [query_weights, positional_factors, \
        rouge1_scores, rouge2_scores, rouge3_scores, rouge4_scores]
    np.savez(data_filename, rouge_data=rouge_data)

plot_type='surface' #contour
if plot_type == 'surface':
    QW, PF = np.meshgrid(rouge_data[0], rouge_data[1])
    Z = np.array(rouge_data[2]).transpose()
    print(QW.shape, PF.shape, Z.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(QW, PF, Z)
    plt.show()

elif plot_type == 'contour':
    qws = np.array(rouge_data[0])
    pfs = np.array(rouge_data[1])
    rouge1_scores = np.array(rouge_data[2])

    plt.contourf(pfs,qws,rouge1_scores)
    plt.ylabel('Query weight')
    plt.xlabel('Decreasing relevance factor')
    plt.colorbar()
    plt.show()
