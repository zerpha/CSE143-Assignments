import math

#probability of sentence using unigram model
def unigram(line, wordTotal, dict):
    prob = 1
    listA = line.split()
    for word in listA:
         prob = prob * (dict[word]/wordTotal)
    prob = prob * (dict['<STOP>'] / wordTotal)
    return prob

#########################################################################
#probability of unigram model using entire data set(list)
def perplexityUni(list, wordTotal, dict, sumofwords):
    sum = 0
    for x in range(0, len(list)):
        listA = list[x].split()
        for word in listA:
            sum = sum + math.log2(dict[word] / wordTotal)
        sum = sum + math.log2(dict['<STOP>'] / wordTotal)
    return (2 ** ((-1) * (sum / sumofwords)))  # perplexity of unigram


###################################################################################
#probability of sentence using bigram model
def bigram(line, dict, dictBi):
    line = '<START> ' + line + ' <STOP>'
    prob = 1
    list = line.split()

    if (list[0], list[1]) in dictBi.keys():
        prob = prob * (dictBi[(list[0], list[1])] / dict['<STOP>'])
    else:
        return 0

    for x in range(1, len(list)-1):
         if (list[x], list[x + 1]) in dictBi.keys():
            prob = prob * (dictBi[(list[x], list[x + 1])]/dict[list[x]])
         else:
            return 0
    return prob
#######################################################################
#probability of bigram model using entire data set(list)
def perplexityBi(list, dict, dictBi, sumofwords):
    sum = 0
    for x in range(0, len(list)):
        line = list[x]
        line = '<START> ' + line + ' <STOP>'
        listA = line.split()

        if (listA[0], listA[1]) in dictBi.keys():
            sum = sum + math.log2(dictBi[(listA[0], listA[1])] / dict['<STOP>'])
        else:
            return -1 #let -1 represent inf perplexity

        for x in range(1, len(listA) - 1):
            if (listA[x], listA[x + 1]) in dictBi.keys():
                sum = sum + math.log2(dictBi[(listA[x], listA[x + 1])] / dict[listA[x]])
            else:
                return -1 #let -1 represent inf perplexity
    return (2 ** ((-1) * (sum / sumofwords)))  # perplexity of bigram


###################################################################################
#probability of sentence using trigram model
def trigram(line, dictBi, dictTri):
    line = '<START> ' + line + ' <STOP>'
    prob = 1
    list = line.split()

    for x in range(0, len(list)-2):
         if (list[x], list[x + 1], list[x + 2]) in dictTri.keys():
            prob = prob * (dictTri[(list[x], list[x + 1], list[x + 2])]/dictBi[(list[x], list[x+1])])
         else:
            return 0
    return prob


###################################################################################
#probability of trigram model using entire data set(list)
def perplexityTri(list, dictBi, dictTri, sumofwords):
    sum = 0
    for x in range(0, len(list)):
        line = list[x]
        line = '<START> ' + line + ' <STOP>'
        listA = line.split()

        for x in range(0, len(listA) - 2):
            if (listA[x], listA[x + 1], listA[x + 2]) in dictTri.keys():
                sum = sum + math.log2(dictTri[(listA[x], listA[x + 1], listA[x + 2])] / dictBi[(listA[x], listA[x+1])])
            else:
                return -1 #let -1 represent inf perplexity
    return (2 ** ((-1) * (sum / sumofwords)))  # perplexity of trigram


###################################################################################
#probability of sentence using smoothing of all three models
def smoothing(line, dict, dictBi, dictTri, wordTotal):
    lamb1 = 0.1
    lamb2 = 0.3
    lamb3 = 0.6
    OVERALL = 1

    line = '<START> ' + line + ' <STOP>'
    list = line.split()

    uni = 1
    bi = 1
    tri = 1

    if (list[0], list[1]) in dictBi.keys():
        bi = bi * (dictBi[(list[0], list[1])] / dict['<STOP>'])  #same amount of stops as starts
        tri = tri * (dictBi[(list[0], list[1])] / dict['<STOP>'])
    else:
        bi = 0
        tri = 0
    uni = uni * (dict[(list[1])] / wordTotal)
    OVERALL = OVERALL * ((lamb1 * uni) + (lamb2 * bi) + (lamb3 * tri))

    for x in range(0, len(list)-2):
         uni = 1
         bi = 1
         tri = 1
         if (list[x], list[x + 1], list[x + 2]) in dictTri.keys():
            tri = tri * (dictTri[(list[x], list[x + 1], list[x + 2])]/dictBi[(list[x], list[x+1])])
         else:
            tri = 0
         if (list[x + 1], list[x + 2]) in dictBi.keys():
            bi = bi * (dictBi[(list[x + 1], list[x + 2])]/dict[list[x + 1]])
         else:
            bi = 0
         uni = uni * (dict[(list[x + 2])] / wordTotal)
         OVERALL = OVERALL * ((lamb1 * uni) + (lamb2 * bi) + (lamb3 * tri))
    return OVERALL


###################################################################################
#perplexity of smoothing
def perplexitySmooth(LIST, dict, dictBi, dictTri, wordTotal, sumofwords):
    lamb1 = 0.3
    lamb2 = 0.3
    lamb3 = 0.4
    sum = 0

    for x in range(0, len(LIST)):
        line = LIST[x]
        line = '<START> ' + line + ' <STOP>'
        list = line.split()

        uni = 1
        bi = 1
        tri = 1

        if (list[0], list[1]) in dictBi.keys():
            bi = bi * (dictBi[(list[0], list[1])] / dict['<STOP>'])  #same amount of stops as starts
            tri = tri * (dictBi[(list[0], list[1])] / dict['<STOP>'])
        else:
            bi = 0 #log(0) = -inf
            tri = 0
        uni = uni * (dict[(list[1])] / wordTotal)
        sum = sum + math.log2((lamb1 * uni) + (lamb2 * bi) + (lamb3 * tri))

        for x in range(0, len(list)-2):
             uni = 1
             bi = 1
             tri = 1
             if (list[x], list[x + 1], list[x + 2]) in dictTri.keys():
                tri = tri * (dictTri[(list[x], list[x + 1], list[x + 2])]/dictBi[(list[x], list[x+1])])
             else:
                tri = 0
             if (list[x + 1], list[x + 2]) in dictBi.keys():
                bi = bi * (dictBi[(list[x + 1], list[x + 2])]/dict[list[x + 1]])
             else:
                bi = 0
             uni = uni * (dict[(list[x + 2])] / wordTotal)
             sum = sum + math.log2((lamb1 * uni) + (lamb2 * bi) + (lamb3 * tri))
    return (2 ** ((-1) * (sum / sumofwords)))


###################################################################################
def main():
    listUni = []    #list of unigrams
    listBT = [] #list used to build bigram and trigram model, has data stored line by line
    listBTcopy = [] #copy of list stored line by line, since other list has later has <STOP> and <START> inserted
    listDev = []  #list of dev data
    listTest = [] #list of test data

    dict = {'<STOP>': 0}
    dictBi = {}
    dictTri = {}
    f1 = open("1b_benchmark.train.tokens", encoding = "utf8")
    f2 = open("1b_benchmark.dev.tokens", encoding="utf8")
    f3 = open("1b_benchmark.test.tokens", encoding="utf8")

    #read in training data
    for line in f1:
        dict['<STOP>'] = dict['<STOP>'] + 1
        listUni.extend(line.split())   #for unigram
        listBT.append(line)  #for bigram, trigram
        listBTcopy.append(line)
    wordTotal = len(listUni) + dict['<STOP>'] #for unigram

    #read in dev data
    for line in f2:
        listDev.append(line)

    #read in test data
    for line in f3:
        listTest.append(line)

    #for bigram and trigram model
    for x in range(0, len(listBT)):
        listBT[x] = '<START> ' + listBT[x] + ' <STOP>'

    #count tokens into dict
    for word in listUni:
        if word in dict.keys():
            dict[word] = dict[word] + 1
        else:
            dict[word] = 1

    #replace token less than 3 as <UNK>
    UNK = 0
    delete = [key for key in dict if dict[key] < 3]
    for key in delete:
        del dict[key]
        UNK = UNK + 1
    dict['<UNK>'] = UNK

    # UNIQUE TOKENS count
    print(len(dict.keys()))

    #insert <UNK> into bigram
    biRes = []
    for sent in listBT:
        listX = sent.split()
        for x in range(0, len(listX)-1):
            str1 = listX[x]
            str2 = listX[x+1]
            if listX[x] != '<START>' and listX[x] not in dict.keys():
                str1 = '<UNK>'
            if listX[x+1] != '<START>' and listX[x+1] not in dict.keys():
                str2 = '<UNK>'
            biRes.append((str1, str2))

    #build bigram dictionary
    for bi in biRes:
        if bi in dictBi.keys():
            dictBi[bi] = dictBi[bi] + 1
        else:
            dictBi[bi] = 1

    #insert <UNK> into trigram
    triRes = []
    for sent in listBT:
        listY = sent.split()
        for x in range(0, len(listY)-2):
            str1 = listY[x]
            str2 = listY[x+1]
            str3 = listY[x+2]
            if listY[x] != '<START>' and listY[x] not in dict.keys():
                str1 = '<UNK>'
            if listY[x + 1] != '<START>' and listY[x + 1] not in dict.keys():
                str2 = '<UNK>'
            if listY[x + 2] != '<START>' and listY[x + 2] not in dict.keys():
                str3 = '<UNK>'
            triRes.append((str1, str2, str3))

    #build trigram dict
    for tri in triRes:
        if tri in dictTri.keys():
            dictTri[tri] = dictTri[tri] + 1
        else:
            dictTri[tri] = 1

    # enter <UNK> into dev list
    listTEMP = []
    for x in range(0, len(listDev)):
        temp = listDev[x].split()
        strtmp = ''
        for x in range(0, len(temp)):
            if temp[x] not in dict.keys():
                strtmp = strtmp + '<UNK> '
            else:
                strtmp = strtmp + temp[x] + ' '
        listTEMP.append(strtmp)
    listDev = listTEMP

# enter <UNK> into listBTcopy list
    listTEMP = []
    for x in range(0, len(listBTcopy)):
        temp = listBTcopy[x].split()
        strtmp = ''
        for x in range(0, len(temp)):
            if temp[x] not in dict.keys():
                strtmp = strtmp + '<UNK> '
            else:
                strtmp = strtmp + temp[x] + ' '
        listTEMP.append(strtmp)
    listBTcopy = listTEMP

#enter <UNK> into test list
    listTEMP = []
    for x in range(0, len(listTest)):
        temp = listTest[x].split()
        strtmp = ''
        for x in range(0, len(temp)):
            if temp[x] not in dict.keys():
                strtmp = strtmp + '<UNK> '
            else:
                strtmp = strtmp + temp[x] + ' '
        listTEMP.append(strtmp)
    listTest = listTEMP

#for the perplexity, find total word count in dev file
    wordTotalDev = 0
    for line in listDev:
        temp = line.split()
        wordTotalDev = wordTotalDev + len(temp) + 1 # + 1 is for the <STOP>

# for the perplexity, find total word count in test file
    wordTotalTest = 0
    for line in listTest:
        temp = line.split()
        wordTotalTest = wordTotalTest + len(temp) + 1 # + 1 is for the <STOP>

# training data
    # for line in listBTcopy:
    #     print(unigram(line, wordTotal, dict))
    # for line in listBTcopy:
    #     print(bigram(line, dict, dictBi))
    # for line in listBTcopy:
    #     print(trigram(line, dictBi, dictTri))
    # for line in listBTcopy:
    #     print(smoothing(line, dict, dictBi, dictTri, wordTotal))

# dev data
#     for line in listDev:
#         print(unigram(line, wordTotal, dict))
#     for line in listDev:
#         print(bigram(line, dict, dictBi))
#     for line in listDev:
#         print(trigram(line, dictBi, dictTri))
#     for line in listDev:
#         print(smoothing(line, dict, dictBi, dictTri, wordTotal))
# test data
#     for line in listTest:
#         print(unigram(line, wordTotal, dict))
#     for line in listTest:
#         print(bigram(line, dict, dictBi))
#     for line in listTest:
#         print(trigram(line, dictBi, dictTri))
#     for line in listTest:
#         print(smoothing(line, dict, dictBi, dictTri, wordTotal))

    # let -1 represent inf perplexity
    print("Perplexity of Unigram with Trainset:",perplexityUni(listBTcopy, wordTotal, dict, wordTotal))
    print("Perplexity of Unigram with Devset:",perplexityUni(listDev, wordTotal, dict, wordTotalDev))
    print("")
    print("Perplexity of Bigram with Trainset:", perplexityBi(listBTcopy, dict, dictBi, wordTotal))
    print("Perplexity of Bigram with Devset:",perplexityBi(listDev, dict, dictBi, wordTotalDev))
    print("")
    print("Perplexity of Trigram with Trainset:", perplexityTri(listBTcopy, dictBi, dictTri, wordTotal))
    print("Perplexity of Trigram with Devset:", perplexityTri(listDev, dictBi, dictTri, wordTotalDev))
    print("")
    print("Perplexity of Smoothing with Trainset:",perplexitySmooth(listBTcopy, dict, dictBi, dictTri, wordTotal, wordTotal))
    print("Perplexity of Smoothing with Devset:", perplexitySmooth(listDev, dict, dictBi, dictTri, wordTotal, wordTotalDev))
    print("")
    print("Perplexity of Unigram with Testset:", perplexityUni(listTest, wordTotal, dict, wordTotalTest))
    print("Perplexity of Bigram with Testset:", perplexityBi(listTest, dict, dictBi, wordTotalTest))
    print("Perplexity of Trigram with Testset:", perplexityTri(listTest, dictBi, dictTri, wordTotalTest))
    print("Perplexity of Smoothing with Testset:", perplexitySmooth(listTest, dict, dictBi, dictTri, wordTotal, wordTotalTest))

    f1.close()
    f2.close()
    f3.close()

main()