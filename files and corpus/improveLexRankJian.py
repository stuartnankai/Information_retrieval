#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Use Daniel's sentences list to calculate the score by using Power iteration.
computes the transition probability for a random walker on a graph to go from any one node to any other node,\
 where all edges come from sufficiently high similarities between nodes.
finalList is the score list'''
import operator
import math
from itertools import combinations
import nltk
import time
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
import nltk.data

'''Only the sentences list is the input, not the article and section list, because I just need to build one graph for
all sentences and do not need to consider where the sentence comes from. The output will be a score list for sentences
without sorted, so the first score is for the first sentence and so on'''
def improveLexRank(articleindices):
    abbreviations = {'dr.': 'doctor', 'mr.': 'mister', 'bro.': 'brother', 'bro': 'brother', 'mrs.': 'mistress',
                     'ms.': 'miss', 'jr.': 'junior', 'sr.': 'senior',
                     'i.e.': 'for example', 'e.g.': 'for example', 'vs.': 'versus'}
    terminators = ['.', '!', '?', '\n']
    wrappers = ['"', "'", ')', ']', '}', ',', ';', '(', '{', '=', '[', '{', ':', '.', "``", "''", "'s", '[edit]', '?',
                '”']

    findStopWords = stopwords.words("english")  # two stopWord list: findStopWords and stopwordsList
    sentenceLength = len(articleindices)  # all of the sentences which have been tokenized by Daniel
    sentenceList = dict()
    orderNum = 0
    interrogative = dict()

    scoreList = []

    tokenList = dict()
    infoScore = dict()
    tempSem = dict()
    stopwordsList = []
    finalWordList = []
    scoreListFile = open("infoscore.txt", 'r')

    with open('infoscore.txt') as f:
        lines = f.read().splitlines()
        scoreList.append(lines)

    print("This is scoreList:", scoreList)

    with open('stopword.txt') as f:
        lines = f.read().splitlines()
        stopwordsList.append(lines)

    # print(wrappers)
    # print("This is stopwordsList:", stopwordsList)

    for i in range(sentenceLength):
        for stop in articleindices[i]:
            if stop in terminators:
                sentences = articleindices[i].split(stop)
                if stop == '?':
                    interrogative[orderNum] = 0
                else:
                    interrogative[orderNum] = 1
                if stop is not '\n':
                    sentenceList[orderNum] = sentences[0].lower() + stop
                    orderNum += 1
                    break
                else:
                    sentenceList[orderNum] = sentences[0].lower() + '.'
                    orderNum += 1
                    break
    # print("This is interrogative:", interrogative)  # We do not need interrogative sentence in the final text
    print("This is sentence dic:")
    print(sentenceList)
    documentsSize = len(sentenceList)  # how many sentences we pick up

    def getInfoScore(sentence):  # Check the sentence contains key words or not.
        score = 0
        for word in sentence:
            if word in scoreList[0]:
                # print("This is scoreList:", scoreList[0])
                score += 1  # weight, we can change it
        return score

    for i in range(documentsSize):

        tokens = nltk.word_tokenize(sentenceList.get(i))  # tokenizer for every sentence
        tempWordList = [word for word in tokens if
                        word not in findStopWords and wrappers and stopwordsList]  # remove the stop word
        # for term in tempWordList:
        infoScore[i] = getInfoScore(tempWordList)
        content = ' '.join(tempWordList)
        content = content.replace("?", '')
        content = content.replace(".", '')
        content = content.replace('"','')
        content = content.replace(',', '')
        content = content.replace('``', '')
        content = content.replace(";",'')
        tempSem[i] = content  # creat a new dict which contains the new sentence without stop word
        stemmer = SnowballStemmer("english")  # find the stem
        for word in tempWordList:
            if word not in wrappers:
                if word not in stopwordsList[0]:
                    finalWordList.append(stemmer.stem(word))
        # print("This is finalWordList:",finalWordList)
        # print("Total words:",len(finalWordList)) # Total number of words
        for x in finalWordList:
            tokenList[x] = finalWordList.count(x)
        # print("This is temSem sample:", tempSem[i])
    N = len(tempSem)  # the new list for all sentences
    average = []
    averageScore = 0
    print("This is the number of sentences:", N)
    # print("This is info score:", infoScore)
    print()
    print("This is final word list: ", finalWordList)
    print("This is the size of final word list:", len(finalWordList))
    # print("This is token List: ", tokenList)
    sorted_List = sorted(tokenList.items(), key=operator.itemgetter(1), reverse=True)[:]  # pick up all
    print("This is sorted list for terms:", sorted_List)
    print()

    def stringToSentence(String):
        listTemp = nltk.re.sub("[^\w]", " ", String).split()
        return listTemp

    def getTfIdf(term, sentence, sentenceList):
        numTf = 0
        numDf = 0
        for word in sentence:
            if word == term:
                numTf += 1
        # print("This is numTF",numTf/len(sentence))

        for num in range(documentsSize):
            if term in sentenceList[num]:
                numDf += 1
        # print("This is numDf:", numDf)
        return float((numTf / len(sentence)) * math.log(documentsSize / numDf))

    def findSimilarity(term, list2):  # compare the word similarity for two sentences
        list = []
        for word2 in list2:
            wordFromList1 = wordnet.synsets(term)
            wordFromList2 = wordnet.synsets(word2)
            # print("This is wordFromList1[0]:", wordFromList1[0])
            # print("This is wordFromList2[0]:", wordFromList2[0])
            if wordFromList1 and wordFromList2:
                s = wordFromList1[0].wup_similarity(wordFromList2[0])
                if s is None:
                    list.append(0)
                else:
                    list.append(s)
        if len(list) == 0:
            return 0
        else:
            return float(max(list))  # return the max number

    tempGroup = list(combinations(range(N), 2))  # compare random two sentences and then calculate the similarity
    similarityValue = dict()
    sentenceWeight = 0.6
    lengthWeight = 1 - sentenceWeight
    similarityThreshold = 1  # We can change. If the similarity is lower than 1, we think they do not have connection

    for i in tempGroup:  # calculate SemSim(A,B) !!!!
        A = i[0]  # the order number of first sentence
        B = i[1]  # the order number of second sentence

        similarityScoreA = []
        similarityScoreB = []
        weightForFirst = []
        weightForSecond = []
        # print("This is tempSem[A]:", tempSem[A])  # fist sentence
        # print("This is tempSem[B]:", tempSem[B])  # second sentence
        temptempA = stringToSentence(tempSem[A])  # String to sentence
        temptempB = stringToSentence(tempSem[B])  # get the second sentence
        for term in temptempA:
            # if term in keyWordList:
            tempATdIdf = getTfIdf(term, temptempA, tempSem)  # calculate the tf-idf
            # print("This is sim value:",findSimilarity(term, temptempB))
            # print("This is if-idf:", tempATdIdf)
            similarityScoreA.append(float(findSimilarity(term, temptempB)) * tempATdIdf)
            weightForFirst.append(tempATdIdf)  # weight for term

        for term in temptempB:
            # if term in keyWordList:
            tempBTdIdf = getTfIdf(term, temptempB, tempSem)
            # print("This is tempBTdIdf:", tempBTdIdf)
            similarityScoreB.append(float(findSimilarity(term, temptempA)) * tempBTdIdf)
            weightForSecond.append(tempBTdIdf)

        if sum(weightForFirst) and sum(weightForSecond) != 0:
            # print("This is sum score A:", sum(similarityScoreA))
            simAB = (sum(similarityScoreA) / sum(weightForFirst)) + (
                sum(similarityScoreB) / sum(weightForSecond))  # !!! Important
        else:
            simAB = 0

        lengthSim = 1 - abs((len(temptempA) - len(temptempB)) / (len(temptempA) + len(temptempB)))

        similarityValue[i] = (interrogative[A] * interrogative[B]) * (
            sentenceWeight * simAB + lengthWeight * lengthSim)  # Most important part
        average.append((interrogative[A] * interrogative[B]) * (sentenceWeight * simAB + lengthWeight * lengthSim))

    finalResult = sorted(similarityValue.items(), key=operator.itemgetter(1), reverse=True)

    print("This is finalResult:", finalResult)
    guessValueC = 0.85  # google used

    def similarityMatrix(finalResult):
        result = [[0 for i in range(N)] for i in range(N)]
        for sen in finalResult:
            # print("This is sen:", sen)
            n = sen[0][0]
            # print("N", n)
            m = sen[0][1]
            # print("M:", m)
            result[n][n] = 0
            result[m][m] = 0
            result[n][m] = sen[1]
            # print("sen[1]:", sen[1])
            result[m][n] = sen[1]
        return result

    similarities = similarityMatrix(finalResult)
    print("This is similarities matrix:", similarities)

    '''This is used as bonus, such as if a sentence is the first sentence of paragraph, we think this sentence is more important than
    others. Or if the sentence contains some key words such as (aim, purpose), we can give some bonus. '''
    numOfZero = average.count(0)
    totalV = 0
    for num in average:
        if num != 0:
            totalV += num
    averageScore = totalV / (len(average) - numOfZero)
    print("This is averageSc:", averageScore)

    # averageScore = (sum(average) - N) / (len(average) - N)  # !!!!!!!!!!!!!!!!!!!!
    # print("This is score:")
    # print("This is averageScore:", averageScore)

    def transitionProbabilities(similarities, similarityThreshold, continuoue):
        probabilities = [[0 for x in range(len(similarities[0]))] for y in range(len(similarities))]
        for i in range(len(similarities)):
            totalNum = 0
            for j in range(len(similarities)):
                text = similarities[i][j]
                if similarities[i][j] > similarityThreshold:
                    if continuoue:
                        probabilities[i][j] = similarities[i][j]
                        totalNum += similarities[i][j]
                    else:
                        probabilities[i][j] = 1
                        totalNum += 1
                else:
                    probabilities[i][j] = 0
            for m in range(len(similarities)):
                if totalNum != 0:
                    probabilities[i][m] = (probabilities[i][m] / totalNum) * guessValueC + (1.0 - guessValueC) * (
                        1.0 / N)
                else:
                    probabilities[i][m] = 0
        return probabilities

    # Build the transition matrix
    transitionProbabilities = transitionProbabilities(similarities, similarityThreshold, True)

    print("This is transition Matirx:", transitionProbabilities)

    def multMatrix(firstMatrix, secondVector):
        if len(firstMatrix) == 0 and len(secondVector) == 0:
            return None
        if len(firstMatrix[0]) != len(secondVector):
            return None
        resultMulti = [[0] for y in range(len(firstMatrix[0]))]
        for i in range(len(firstMatrix[0])):
            for j in range(len(secondVector[0])):
                multiNum = 0
                for k in range(len(secondVector)):
                    multiNum += firstMatrix[i][k] * secondVector[k][0]
                resultMulti[i][0] = multiNum
        # print("Multi is running:", resultMulti)
        return resultMulti

    def powerIteration(transitionProbabilities, size, epsilon, maxIteration):
        # currentMatrix = transposeMatrix(transitionProbabilities)
        currentMatrix = transitionProbabilities
        currentVector = [[0.0] for y in range(size)]
        for i in range(size):
            if i == 0:
                currentVector[i][0] = 1
            else:
                currentVector[i][0] = 0
        for i in range(maxIteration):
            previousVector = currentVector
            currentVector = multMatrix(currentMatrix, currentVector)
            error = 0
            # for j in range(size):
            #     error += math.pow(float(currentVector[j][0]) - float(previousVector[j][0]), 2)
            # if error < math.pow(epsilon, 2):
            #     break
            for j in range(size):
                error += currentVector[j][0] - previousVector[j][0]
            if error < epsilon:
                break
        result1 = dict()
        print("This is currentVector:", currentVector)
        for i in range(size):
            result1[i] = currentVector[i][0]
        return result1

    rankings = powerIteration(transitionProbabilities, N, 0.0001, 100)

    autoSum = sorted(rankings.items(), key=operator.itemgetter(1), reverse=True)
    # print("This is autoSum:", autoSum)
    for item in autoSum:
        numOfSentence = item[0]
        saveText.write(sentenceList[numOfSentence].capitalize())
        saveText.write('\n')
        print("After Sum:", sentenceList[numOfSentence].capitalize())
    # for i in range(len(rankings)):
    #     rankings[i] = rankings.get(i) + infoScore[i] * averageScore
    finalList = []
    for i in range(len(rankings)):
        finalList.append(rankings[i])
    # print("This is list:", finalList)
    return finalList


def readFirstLine():
    readText = open('sample1.txt', 'r')
    sentences = []
    checkedPar = False
    readAll = False
    terminators = ['.', '!', '?', '\n']
    while readAll is False:
        findFirstS = readText.readline()
        if findFirstS != None:
            findFirstS.replace('\n', '.')
            if checkedPar is not True:  # the paragraph is checked or not
                checkedPar = True
                for stop in findFirstS:
                    if stop in terminators:
                        sentencesTemp = findFirstS.split(stop)
                        if stop is not '\n':
                            sentences.append(sentencesTemp[0].lower() + stop)
                            break
                        else:
                            sentences.append(sentencesTemp[0].lower() + '.')
                            break
            elif len(findFirstS) == 0:
                break
            elif findFirstS.count('\n') == len(findFirstS):  # go to the next paragraph
                checkedPar = False
        else:
            readAll = True
            break
    return sentences


if __name__ == '__main__':
    tStart = time.time()
    saveText = open('saveT.txt', 'w')
    # If we use all sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    print("")
    fp = open("sample.txt",'r')
    data = fp.read()
    # pick up the first sentence
    firstPList = readFirstLine()
    print("This is list for firstPList:",firstPList)
    #pick up all sentences
    findSentence = tokenizer.tokenize(data)
    print("This is findSentence:", findSentence)
    result = improveLexRank(articleindices=findSentence)
    print("This is final result:", result) 
    tEnd = time.time()
    print("Total time is :", tEnd - tStart)
