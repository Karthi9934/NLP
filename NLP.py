"""
==========================================
NLP 
==========================================
"""
import os
import warnings
from collections import Counter,defaultdict
warnings.filterwarnings('ignore')
#Phonetic packages
import re
import en_core_web_sm
nlp = en_core_web_sm.load()
from word2number import w2n
#FAQ Section
import nltk.tokenize.punkt
import nltk.stem.snowball
import string
from difflib import SequenceMatcher
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
import pandas as pd
import unicode

# Path declaration
path =''
os.chdir(path)

############################ READING CSV FILES ################################
# Reading the Master Reference File
ref_file = pd.read_csv('Master_Reference_File.csv',keep_default_na=False,na_values = "")
ref_file = ref_file.fillna(0)
ref_file['YEAR'] = ref_file['YEAR'].map(int)
# Reading the FAQ CSV file
faq = pd.read_csv("FAQ.csv")
# Reading Lookup words CSV file
kpiLookUp = pd.read_csv('lookUp.csv')

###################### ASSIGNING READ DATA TO VARIABLES #######################
lookUpDict = dict(map(lambda x,y : (x,y.split(',')),kpiLookUp['Key'],kpiLookUp['Value']))
# Dictionary of Dimensions
Dimensions = dict(zip(list(ref_file),map(lambda x: ref_file[x].unique(),list(ref_file))))
# Testing Words List
dimensionAll = np.array(ref_file).flatten().tolist()
dimensionAll = [x for x in dimensionAll if type(x) == str]
# For KPI's which are present in Multiple Columns eg: MACO is in drivers and Intents
multiKPI = [k for k,v in Counter(dimensionAll).items() if v > 1]
multiKPIMap = defaultdict(list)
[multiKPIMap[x].append(y) for x in multiKPI for y in Dimensions.keys() if x in Dimensions[y]]
multiKPIMap = dict(multiKPIMap)
# Creating testing having every single word of master reference for auto-correction
testing = sum([x.split() for x in dimensionAll],[])
# Extending lookUpFiles into the testing for AutoCorrect
[testing.extend(i.split()) for t in lookUpDict.values() for i in t]
# Create tokenizer and stemmer
tokenizer = nltk.tokenize.PunktSentenceTokenizer()
stemmer = nltk.stem.snowball.SnowballStemmer('english')
# Get default English stopwords and extend with punctuation and other used words
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.extend(list(pd.read_csv("stopWords.csv",header=None)[0]))

################################## FUNCTIONS ##################################
#FAQ Function
def faqCheck(ques):
    """ Checks if the query is a FAQ.
    
    This function is executed from :py:func:`out_int` function to check if the query is a FAQ. 
    FAQ.csv file is read and saved as a dataframe in :py:mod:`faq`. The ques is 
    compared with the queries(value column) of faq(dataframe) using `SequenceMatcher <https://docs.python.org/2/library/difflib.html>`_ 
    function which gives a score.
    
    .. csv-table:: faq (dataframe)
        :header: "value","response"
        :widths: 50,70
        
        "How is MROI calculated?","MROI is calculated as a ratio of MACO to Spend"
        "...","..."

    Parameters
    ----------
    ques : string
        The text string with which ML is called with is passed as argument.
        
    Returns
    -------
    Response : string
        If the score is above threshold for a query of faq dataframe, the response 
        related to the matched query is returned.
    None : None
        If none of the queries is matched, then the text is not a FAQ, we return 
        None in that case.
        
    """
    for i,j in enumerate(faq['value']):
        if (SequenceMatcher(None,ques.lower(),j.lower()).ratio()) > 0.85:
            return faq['Response'][i]
    else:
        return None

# Look Up Functions to check pairs of words in look up dictionary
def _lookup_words(pairs1):
    """ Checks if a word combination is in lookUp Synonym Dictionary
    
    This function is executed from :py:func:`intent` function. Checks if the possible combinations of words 
    of the query is a synonym to a KPI or Dimension saved in :py:mod:`lookUpDict` (Dictionary) values.
    
    Parameters
    ----------
    pairs1 : list of strings
        A list of strings having all possible sequential combinations of words of the query
    
    Returns
    -------
    new_words : list of strings
        A list of strings having all the KPI or Dimensions to the synonym matched from pairs1 list with 
        to the values of :py:mod:`lookUpDict` Dictionary.
        
    """
    new_words = [ k for word in pairs1 for k,v in lookUpDict.iteritems() for j in v if word == j]
    return new_words

# Association Code
def numExtract(someMatch):
    """ Called from :py:func:`Association` 
    
    Transforming the match found in association into a usable format
    
    Parameters
    ----------
    someMatch : String
        Extracted format of string from Association function
        
    Eg : "GREATER THAN 1","PAST 2 YEARS"
    
    Returns
    -------
    Tuple : A tuple of a string and a number
        Transforms the input string into specific format
        
    Eg : ("GREATER THAN",1),("PAST YEARS",2)
    
    """
    numbers = re.findall("\d+\.?\d*",someMatch)
    for num in numbers: someMatch = someMatch.replace(num,"",1)
    numProd = np.prod(np.array(map(float,numbers)))
    return (str(" ".join(someMatch.split())),numProd)
    
def Association(text):
    """ Finds Associations in the text
    
    Called from :py:func:`intent` function. Finds the following objects if present in the query:
        * Number Association : Uses regular expressions.
        * Time Reference Association: Uses `en_core_web_sm <https://spacy.io/usage/models>`_ model from SpaCy for NLP to detect time referred objects in query.
    
    Parameters
    ----------
    text : string
        The query string asked is passed as argument.
        
    Eg : "COUNTRIES HAVING MROI GREATER THAN 1 FOR LAST 2 YEARS"
    
    Returns
    -------
    List : A list of tuples
        List of tuples giving all the associations found in the query.
        
    Eg : [("GREATER THAN",1),("LAST YEARS",2)]
    
    """
    # Number Comparison Association
    pattern = "GREATER THAN \d+\.?\d*\s?\d*|LESS[ER]* THAN \d+\.?\d*\s?\d*|MORE THAN \d+\.?\d*\s?\d*|EQUAL TO \d+\.?\d*\s?\d*|SMALL[ER]* THAN \d+\.?\d*\s?\d*|ABOVE \d+\.?\d*\s?\d*|BELOW \d+\.?\d*\s?\d*|NOT EQUAL TO \d+\.?\d*\s?\d*"
    numComparison = [numExtract(match) for match in re.findall(pattern,text)]
    # Time Association
    doc = nlp(unicode(text))
    timeReference = [e.string for e in doc.ents if 'DATE'==e.label_]
    timeReference = [numExtract(match) for match in timeReference]
    timeReference = [x for x in timeReference if x[1] not in Dimensions['YEAR']]
    return numComparison+timeReference

def sequnceCombinations(strLis,n):
    """ Sequential combinations of words of the query
    
    Parameters
    ----------
    strLis : List
        A list of all the words of the query
        
    n : Integer
        Maximum length of word combinations 
        
    Eg : ['WHAT', 'IS', 'NET', 'REVENUE', 'OF', 'BRAZIL']
    
    
    Returns
    -------
    List : A list of strings (word combinations)
        A list of possible sequential combinations of the words of query.
        
    Eg : if n is 3 then all combinations from single word upto 3 words will be returned as: 
         ['WHAT IS NET', 'REVENUE OF', 'WHAT', 'IS NET REVENUE', 'REVENUE', 'WHAT IS', 'IS', 'OF BRAZIL', 
         'NET REVENUE', 'NET REVENUE OF', 'OF', 'IS NET', 'NET', 'REVENUE OF BRAZIL', 'BRAZIL']
    
    """
    seqCombs = [" ".join(strLis[i:i+j]) for i in range(len(strLis)) for j in range(1,n+1)]
    return list(set(seqCombs))

# Intent and slot classification
def intent(text):
    """ Extracts useful entities from the query string
    
    Called from :py:func:`textAnalyse`. Does the main entity mapping from Master_Reference(Dimensions Dictionary) file.
    Calls the various functions like :py:func:`Association`, :py:func:`sequenceCombinations`, :py:func:`_lookup_words`
    to extract out all possible entities out of the text
    
    Parameters
    ----------
    text : string
        The query string asked is passed as argument.
    Eg : "WHAT IS NET REVENUE OF BRAZIL"
        
    Returns
    -------
    mappingDict : Dictionary
        The matched entities from the text are the values of the dictionary while the keys are the column of Master reference file 
        to which they belong to.
    Eg : {"INTENT" : "NET REVENUE","COUNTRY" : "BRAZIL"}
    
    """
    text = wordToNumText(text)
    textList = text.split()
    # Extracting out Quarter mention in text
    qPattern = "QUARTER\s*\d"
    qList = re.findall(qPattern,text)
    # Separating the numbers in a list
    numbers = re.findall("\d+\.?\d*",text)
    #numbers = [int(i) for i in textList if i.isdigit() == True]
    # Taking the years out in a separate list
    yearListStr = [str(i) for i in Dimensions['YEAR']]
    year = [int(i) for i in numbers if i in yearListStr]
    # Numbers which are not years in a list
    numberNonYear = [float(i) for i in numbers if i not in yearListStr]
    # Rest of the words of the query
    words = [i for i in textList if i not in numbers]
    #words = [i for i in textList if i.isdigit() == False]
    numAssociation = []
    if len(numberNonYear) >= 1: # Do Association only if we have a number in query which is not year
        numAssociation = Association(text)
    # Taking all possible sequential combinations (upto 4 words) of the query words in a list
    pairCombs = sequnceCombinations(words,4)
    # Pairs which have a direct match in Master Reference
    matchedPairs = [p for key in Dimensions.keys() for p in pairCombs if p in Dimensions[key]]
    # Solving the Conflict of similar matched entities EG: "Spend Composition" query will have "spend" and "spend composition" both
    keyWords = [key for key in matchedPairs if text.count(key) == " ".join(matchedPairs).count(key)]
    # Getting the unmatched keys in another list
    unmatchedKeys = [key for key in matchedPairs if key not in keyWords]
    # Replace all the keyWords from original Text and check if unmatched keyWords are there in text
    #if keyWords:
    text1 = text
    for key in keyWords: text1 = text1.replace(key,"")
    keyWords += [key for key in unmatchedKeys if key in text1]
    # Getting the unmatched pairs out
    unmatchedPairs = [p for p in pairCombs if p not in matchedPairs]
    # Checking the unmatched pairs and quarter list in lookUp words for synonyms
    # split by "," For case of Global Brands synonyms [budweiser,corona,stella artois]
    lookUpMatch = sum([x.split(",") for x in _lookup_words(unmatchedPairs)],_lookup_words(qList))
    # Collecting all keyWords
    keyWords += lookUpMatch
    keyWords = list(set(keyWords))
    # Q1 Q3 or Quarter 1 Quarter 3 separated with spaces
    qList1 = [re.match('Q\d',key).group() for key in keyWords if re.match('Q\d',key) != None]
    if len(qList1) > 1 and "".join(sorted(qList1)) in Dimensions['TIME PERIOD'] and "AND" not in text:
        keyWords.append("".join(sorted(qList1)))
        [keyWords.remove(i) for i in qList1]
    keyWords += year
    # Taking Brands out in case of global brands
    if 'GLOBAL BRANDS' in text or 'GLOBAL BRAND' in text:
        keyWords = [i for i in keyWords if i not in ['BRANDS','BRAND']]
    # Final Mapping of the entities 
    mappedEntity = [(i,j) for i in Dimensions.keys() for j in keyWords if j in Dimensions[i]]
    mappingDict = defaultdict(list)
    [mappingDict[i[0]].append(i[1]) for i in mappedEntity]
    if len(numberNonYear) >= 1:
        mappingDict['FIELD_VALUES'] = dict(i for i in numAssociation)
        mappingDict['FIELD_VALUES'] = {k:v for k,v in mappingDict['FIELD_VALUES'].items() if 'Q' not in k}
        if mappingDict['FIELD_VALUES'] == {}: mappingDict['FIELD_VALUES'] = None
    return dict(mappingDict)

def wordToNumText(text):
    """ Convert number words to number 
    
    Called from :py:func:`textAnalyse`. Uses `word2number <https://pypi.org/project/word2number/>`_ package.
    Replaces the number words from converted numbers and returns joined list 
    
    Parameters
    ----------
    text : string
        The query string asked is passed as argument.
    Eg : "COUNTRIES HAVING MROI GREATER THAN ONE"
    
    Returns
    -------
    string : Corrected string
        Query with replaced number conversion
    Eg : "COUNTRIES HAVING MROI GREATER THAN 1"
        
    """
    o = []
    for word in text.split():
        try:
            o += [str(w2n.word_to_num(word))]
        except ValueError:
            o += [word]
    return " ".join(o)

# Text Analyse Function Code
def textAnalyse(text):
    """ Calls :py:func:`AutoCorrect` and :py:func:`intent` functions
    
    Called from :py:func:`out_int` function. Flow of the function is as follows:
        * Calls AutoCorrect function with base threshold (0.8).
        *if still a word is remained uncorrected then again calls AutoCorrect function with a reduced
        threshold of 0.7.
        * With the final corrected string calls intent function and gets the mapped
        entities
    
    Parameters
    ----------
    text : string
        The query string asked is passed as argument.
    
    Returns
    -------
    kpiDict : Dictionary
        Returns the dictionary returned by :py:func:`intent` function on the corrected query, with two new 
        keys representing a corrected question and the corrections made by AutoCorrect function.
  
    """
    text = text.upper().replace("'s","").replace('"',"").replace("POINT","")
    text = wordToNumText(text)
    ques,uncorrected,metaDict = AutoCorrect(text)
    if uncorrected:
        corrList,uncorrList,metaDict2 = AutoCorrect(uncorrected,0.7)
        for i,j in metaDict2.items(): ques = ques.replace(i,j)
        metaDict.update(metaDict2)
        kpiDict = intent(ques)
        kpiDict['META_DATA'] = metaDict
        kpiDict['NEW_QUES'] = [ques]
    else:
        kpiDict = intent(ques)
        if metaDict: kpiDict['META_DATA'] = metaDict
    return kpiDict

def out_int(text):
    """ Main function called from input function. 
    
    Flow of the function is as follows:
        * First checks if the query is a FAQ by calling the :py:func:`faqCheck` function,
        If yes then returns its related response 
        * If the query is not a FAQ, then calls :py:func:`textAnalyse` function, and returns a dictionary of
        mapped entities
    
    Parameters
    ----------
    text : string
        The query string asked is passed as argument.
    
    Returns
    -------
    kpiDict : Dictionary
        Returns the final dictionary with all the keys possible, These keys will have values
        if a entity from query is mapped to that key, otherwise value will be None.
    
    """
    text = text.upper()
    faq_response = faqCheck(text)
    if faq_response != None:
        Dimen = ["KEY","RESPONSE"]
        dict_var = {}
        dict_var = {key: None for key in Dimen}
        # Creation of Dict for json
        dict_var['KEY']= ['FAQ']
        dict_var['RESPONSE'] = faq_response
        # Json
        output = {
            #"OBJECTIVE": obj,
            "INTENT": dict_var['KEY'],
            "RESPONSE":dict_var["RESPONSE"]
              }
    else:
        master = textAnalyse(text)
        Dimen = ["ZONE", "KEYS", "COUNTRY", "VEHICLE","SUB VEHICLE", "BRAND", 
                 "TIME PERIOD", "YEAR", "DRIVER" ,"INTENT" , "YSTATUS","FIELDS",
                 "FIELD_VALUES", "MONTH","META_DATA","NEW_QUES","CONFLICT"]
        dict_var = {}
        dict_var = {key: None for key in Dimen}
        # Creation of Dict for json
        for key, value in master.iteritems():
            dict_var[key] = value
        if dict_var['KEYS']:
            dict_var['KEYS'] = [str(" ".join(map(lambda y: lemmatizer.lemmatize(y.lower()).upper(),x.split()))) for x in dict_var['KEYS']]
        # For adding GLOBAL in fields if GLOBAL <KPI> is asked but not GLOBAL BRANDS
        if 'GLOBAL' in text.upper() and lemmatizer.lemmatize(re.search('GLOBAL \S*',text.upper()).group().split()[1]) != 'BRAND':
            if dict_var['FIELDS']: dict_var['FIELDS'].append('GLOBAL')
            else: dict_var['FIELDS'] = ['GLOBAL']
        if 'MARGINAL CONTRIBUTION' in text.upper() and text.upper().count("CONTRIBUTION") < 2:
            dict_var['FIELDS'].remove("CONTRIBUTION")
            if dict_var['FIELDS'] == []: dict_var['FIELDS'] = None
        masterList = {k:v for k,v in master.items() if type(v) == list}
        conflictList = [x for x in multiKPI if x in sum(masterList.values(),[])]
        if conflictList:
            conf = conflictList[0] 
            dict_var['CONFLICT'] = {conf: [multiKPIMap[conf][0],multiKPIMap[conf][1]]}
            dict_var[multiKPIMap[conf][0]].remove(conf)
            dict_var[multiKPIMap[conf][1]].remove(conf)
            if dict_var[multiKPIMap[conf][0]] ==[]: dict_var[multiKPIMap[conf][0]] = None
            if dict_var[multiKPIMap[conf][1]] ==[]: dict_var[multiKPIMap[conf][1]] = None
        
        # Json
        output = {
            #"OBJECTIVE": obj,
            "INTENT": dict_var['INTENT'],
    	      "slots": {
                        "ZONE": dict_var["ZONE"],
                        "KEYS": dict_var["KEYS"],
                        "VEHICLE": dict_var["VEHICLE"],
                        "Sub_Vehicle": dict_var['SUB VEHICLE'],
                        "BRAND": dict_var['BRAND'],
                        "Country": dict_var['COUNTRY'],
                        "YSTATUS": dict_var["YSTATUS"],
                        "YEAR": dict_var["YEAR"],
                        "DRIVER": dict_var["DRIVER"],
                        "TIME PERIOD": dict_var["TIME PERIOD"],
                        "FIELDS":dict_var['FIELDS'],
                        "FIELD_VALUES":dict_var['FIELD_VALUES'],
                        "MONTH":dict_var['MONTH'],
                        "META_DATA":dict_var['META_DATA'],
                        "NEW_QUES":dict_var['NEW_QUES'],
                        "CONFLICT":dict_var['CONFLICT']
    				  	}
        		}
    return output

