import nltk.data
from nltk import pos_tag,word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import urllib.parse
import urllib.request
import simplejson
import os
import sys
import time
import glob
import codecs
import unicodedata
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

CHATNOIR = 'http://webis15.medien.uni-weimar.de/proxy/chatnoir/batchquery.json'
CLUEWEB = 'http://webis15.medien.uni-weimar.de/proxy/clueweb/id/'
SnippetChatNoir = 'http://webis15.medien.uni-weimar.de/chatnoir/snippet?'

#ChatNoir Web Document Access (url open)
CNWebDocAccess = 'http://webis15.medien.uni-weimar.de/chatnoir/clueweb?'

SuspDir = os.getcwd() + '\\SuspDir'
OutDir = os.getcwd() + '\\OutDir'
Token = 'HonLerjyewtAvroajCharnoognectunn'

sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")
stopWord = stopwords.words('english')
RegTok = RegexpTokenizer(r'\w+')
numbers = ["0","1","2","3","4","5","6","7","8","9"]

"""         Global variables of approaching Plagiarism Detection            """
#The sentences of a paragraph
k = 8
#The max words allowed in a query sentence
MaxWordsSentence = 18
#The minimum value of ranked query-sentences which will be sent to ChatNoir
MinRankedQS = 10
#Max results focus returned from a query search on ChatNoir
MaxReturnedResult = 6
#Percentage of the query on Snippet to allow if the result will be downloaded
PercentageQueryOnSnippet = 78


class PlagiarismDetection:

    def process(self, Suspdoc, OutDir, Token):

        # Number of queries submitted
        NrQueries = 0
        # Number of web pages downloaded
        NrWebPage = 0
        # Number of queries until the first actual source is found
        NrQueriesFS = 0
        # Number of downloads until the first actual source is downloaded
        NrWebPageFS = 0

        # auxiliary variable for counting of NrQueriesFS & NrWebPageFS
        lock = 0

        #List of Actual Retrieval Sources Id Documents, with near-duplicate Documents
        DActualSource = []

        #List of unique Retrieval Sources, no near-duplicate Documents
        Dret = []

        #List of Id Sources at json
        ListofSourceId = []
        Dsrc = []
        Dsrc_data = []

        # Extract the ID and initiate the log writer.
        self.init(Suspdoc, OutDir)

        # Read and tokenize the suspicious document.
        text = self.read_file(Suspdoc)

        # Extract parts from the suspicious document
        SentOfSuspdoc = sent_detector.tokenize(text.strip())

        k_step = 1
        Max_rank_sentence = 0
        query=""

        cntofSent = 0 #counter of Sentences
        #  each sentence or part
        for each_sentence in SentOfSuspdoc:
            cntofSent+=1

            # Remove from each sentence the punctuation (.,?#@!%^&*- etc)
            words = RegTok.tokenize(each_sentence)

            # Remove stopwords
            words =  [each_word  for each_word in words if each_word not in stopWord]

            #========== Pos Tagging and Ranking Sentences ==================#
            pos_words = pos_tag(words)

            #reduce the length of the sentence, below MaxWordsSentence words
            word_length = 3
            while len(pos_words) > MaxWordsSentence:
                pos_words =  [each_word  for each_word in pos_words if len(each_word[0]) > word_length or len(each_word[0]) == 1 or  each_word[1]=="NNP" or each_word[1] == "NNPS"]

                if(word_length > 8):
                    i=0
                    j=0
                    L=[]
                    for each_word in pos_words:
                        if (i % 2) == 0:
                            L.append(each_word)
                            j+=1
                        i+=1
                    pos_words = L
                    break;

                word_length += 1

            rank_sentence = 0
            for pw in pos_words:
                if len(pw[0]) >= 8:
                    rank_sentence += 1
                if len(pw[0]) >= 6 and len(pw[0]) < 8:
                    rank_sentence += 0.5
                if pw[1] =="NNP" or pw[1] == "NNPS":
                    rank_sentence += 1
            #Sum +=rank_sentence
            #c +=1
            #print("====",rank_sentence)

            # from the k sentences, the most ranked sentence will be sent to ChatNoir
            if k_step == 1:
                Max_rank_sentence = rank_sentence
                # query = make sentence with the union of words of each sentence
                query=""
                for w in pos_words:
                    query = query + w[0] + " "
                query = query[:-1]

            if rank_sentence > Max_rank_sentence:
                Max_rank_sentence = rank_sentence
                # query = make sentence with the union of words of each sentence
                query=""
                for w in pos_words:
                    query = query + w[0] + " "
                query = query[:-1]


            if cntofSent < len(SentOfSuspdoc):
                if k_step != k:
                    k_step += 1
                    continue;
            #print("k-step :",k_step)
            k_step = 1;
            Max_rank_sentence = 0;

            print("Query ====> ",query)
            if(query == ""):
                continue;

            #Send the query request to ChatNoir
            results = self.pose_query(query, Token)

            NrQueries = NrQueries + 1
            # Log the query event.
            self.log(query)

            if results["chatnoir-batch-results"][0]["results"] == 0:
                continue;  # The query returned no results, go to the next query.

            #/////////////////////////////////////////////////////////////////////////#
            #   Downloads the first n-ranked result from a given result list.

            #formating the query request for snippet, ex: query=Family+of+Barack+Obama
            queryWords = word_tokenize(query)
            requestSnippQuery=""
            for qW in queryWords:
                requestSnippQuery = requestSnippQuery + qW + '+'
            requestSnippQuery = requestSnippQuery[:-1]

            for i in range(0,MaxReturnedResult):
                # the returned list of query search could be less than MaxReturnedResult
                try:
                    result = results["chatnoir-batch-results"][0]["result-data"][i]
                except:#IndexError: list index out of range
                    break;#there ara no ranked list results, go to the next query

                document_id = str(result["longid"])

                request = urllib.request.Request(SnippetChatNoir + "id=" +str(document_id) + "&query=" + str(requestSnippQuery) + "&length=500")
                request.add_header("Accept", "application/json")
                request.get_method = lambda: 'GET'

                try:
                    response =  urllib.request.urlopen(request)
                    download_snippet = simplejson.loads(response.read())
                    response.close()
                except urllib.request.HTTPError as e:
                    error_message = e.read()
                    print ( sys.stderr, error_message )
                    sys.exit(1)
                except UnicodeEncodeError:
                    req = (SnippetChatNoir + "id=" +str(document_id) + "&query=" + str(requestSnippQuery) + "&length=500").encode('ascii', 'ignore').decode('ascii')

                    request = urllib.request.Request(req)
                    request.add_header("Accept", "application/json")
                    request.get_method = lambda: 'GET'

                    response =  urllib.request.urlopen(request)
                    download_snippet = simplejson.loads(response.read())
                    response.close()

                snippetWords = word_tokenize(str(download_snippet["snippet"]))

                #count if there is at least one appearance of each query words on snippet
                count_queryWords_onSnippet = 0
                for qW in queryWords:
                    for sW in snippetWords:
                        if qW == sW:
                            count_queryWords_onSnippet +=1
                            break;

                #percentage of the query on Snippet
                pqs = float(count_queryWords_onSnippet*100/len(queryWords))

                #//////////////////////////////////////////////////////////////////
                #if the result is not relevant enough, do not allow to download url
                if pqs < PercentageQueryOnSnippet:
                    continue; #go to next result

                document_url = result["url"]


                foundNearDuplicateDoc = False

                for uret in Dret:
                    if uret[0] == document_id:
                        foundNearDuplicateDoc = True
                        break;

                    if self.Divergenceof2Nr(uret[2],result["characters"]) > 0.87:
                        if SequenceMatcher(None,uret[1],str(result["title"])).ratio() > 0.87:
                            if self.Divergenceof2Nr(uret[3],result["sentences"])>=0.92 and self.Divergenceof2Nr(uret[4],result["words"])>=0.92 and self.Divergenceof2Nr(uret[5],result["syllables"])>=0.92:
                                foundNearDuplicateDoc = True
                                break;

                if foundNearDuplicateDoc == False:
                    Dret.append((document_id, str(result["title"]),result["characters"], result["sentences"],result["words"], result["syllables"]))

                download = self.download_result(document_id, Token)
                NrWebPage = NrWebPage + 1

                if lock == 0 :
                    NrQueriesFS = NrQueries
                    NrWebPageFS = NrWebPage
                # Check the text alignment oracle's verdict about the download.
                if self.check_oracle(download) == 1 :
                    lock=1
                    DActualSource.append( document_id )

                # Log the download event.
                self.log(document_url)
                print("Retrieval Document was placed to log file")



        timeStart = time.time()
        #////////////         Performance Measures        ///////////////
        print("########################\nCalculating Performance Measures")
        #reading the json plagiarism file of Suspicious Document
        json_file =  ''.join([Suspdoc[:-4], '.json'])
        data=open(json_file).read()
        json_data = simplejson.loads(data)

        #Extracting and putting to a List the ID numbers from JSON file of Suspicious Document (Source Documents)
        IDItemsOnJson = len(json_data["plagiarism"])
        templist1 = [] #for numbers ID
        templist2 = [] #for string ID
        for item in range(0,IDItemsOnJson):
            SourceID = json_data["plagiarism"][item]["source-url"]

            if len(SourceID) < 12:
                continue; #go to the next Id plagiarism, cause there is no source-url

            j=1
            while SourceID[-j] in numbers:
                j += 1
            j-=1

            if j > 2:
                templist1.append(SourceID[-j:])
            else:
                j=1
                while SourceID[-j] != '?':
                    j += 1
                j-=1
                templist2.append(SourceID[-j:])
        Dsrc = templist1 + templist2

        positionDeleteUnauthorized=[]
        counterofDsrc=0
        for dsrc in Dsrc:
            #Access web page only for the sources without Id:
            #ex with Id: url = http://webis15.medien.uni-weimar.de/chatnoir/clueweb?id=100015916163&token=HonLerjyewtAvroajCharnoognectunn
            #ex without Id: url = http://webis15.medien.uni-weimar.de/chatnoir/clueweb?href=http%3A%2F%2Fwww.mitchell.edu%2FCurrent%2520Students%2Flibrarytech%2FLITSevents.html&token=HonLerjyewtAvroajCharnoognectunn
            if dsrc[-1] in numbers:
                #Dsrc_data.append( ("null",-1,-1) )
                url = CNWebDocAccess + "id=" + dsrc + "&token=" + Token
                try:
                    html_response =  urllib.request.urlopen(url)
                except:#mostly for-> urllib.error.HTTPError: HTTP Error 401: Unauthorized
                    Dsrc_data.append( ("null",-1,-1) )
                    positionDeleteUnauthorized.append(counterofDsrc)
                    counterofDsrc+=1
                    continue;
                soupSrc = BeautifulSoup(html_response)
                text = soupSrc.get_text().strip()
                sentencesSrc = sent_detector.tokenize(text)
                wordsSrc = word_tokenize(text)

                wordsFirstSent = word_tokenize(sentencesSrc[0][:100])
                firstsentenceSrc=""
                for w in wordsFirstSent:
                    firstsentenceSrc = firstsentenceSrc + w + " "

                Dsrc_data.append( (firstsentenceSrc,len(sentencesSrc),len(wordsSrc)) )
            else:
                url = CNWebDocAccess + dsrc + "&token=" + Token
                try:
                    html_response =  urllib.request.urlopen(url)
                except:#mostly for-> urllib.error.HTTPError: HTTP Error 401: Unauthorized
                    Dsrc_data.append( ("null",-1,-1) )
                    positionDeleteUnauthorized.append(counterofDsrc)
                    counterofDsrc+=1
                    continue;
                soupSrc = BeautifulSoup(html_response)
                text = soupSrc.get_text().strip()
                sentencesSrc = sent_detector.tokenize(text)
                wordsSrc = word_tokenize(text)

                wordsFirstSent = word_tokenize(sentencesSrc[0][:100])
                firstsentenceSrc=""
                for w in wordsFirstSent:
                    firstsentenceSrc = firstsentenceSrc + w + " "

                Dsrc_data.append( (firstsentenceSrc,len(sentencesSrc),len(wordsSrc)) )
            counterofDsrc+=1

        #Delete the founded Ids of Unauthorized document
        cnt = 0
        for posIdDel in positionDeleteUnauthorized:
            if cnt == 0:
                del Dsrc[posIdDel]
                del Dsrc_data[posIdDel]
                cnt+=1
            else:
                del Dsrc[posIdDel-cnt]
                del Dsrc_data[posIdDel-cnt]
                cnt+=1

        #//////////       Compute Precision & Recall      //////////////
        restOfPositiveSources = []
        Nr_Dret_AND_Dsrc = 0 # Number of intersection uniqueDret ^ uniqueDsrc
        Nr_Recall_Calculation  = 0
        Nr_Precision_Calculation = 0

        lengthOfDsrc = len(Dsrc)
        lengthOfDret = len(Dret)



        retCn=0 # Dret counter
        positionIdDelete=[]
        for ret in Dret:

            retAccess = False
            sentencesRet=0
            wordsRet=0
            firstsentenceRet=""

            srcCn=0 #Dsrc counter
            for src in Dsrc:
                if ret[0] == src:      #if the IDs are the same
                    Nr_Dret_AND_Dsrc += 1
                    del Dsrc[srcCn]
                    del Dsrc_data[srcCn]
                    positionIdDelete.append(retCn)
                    break;#go to the next retrieval document
                else:
                    if retAccess == False:
                        url = CNWebDocAccess + "id=" + ret[0] + "&token=" + Token

                        html_response =  urllib.request.urlopen(url)
                        soupRet = BeautifulSoup(html_response)

                        text=soupRet.get_text().strip()
                        sentencesRet = sent_detector.tokenize(text)
                        wordsRet = word_tokenize(text)

                        wordsFirstSent = word_tokenize(sentencesRet[0][:100])

                        firstsentenceRet=""
                        for w in wordsFirstSent:
                            firstsentenceRet = firstsentenceRet + w + " "

                        retAccess = True

                    similarityofSents = SequenceMatcher(None,firstsentenceRet,Dsrc_data[srcCn][0]).ratio()
                    if similarityofSents > 0.87:
                        if self.Divergenceof2Nr(len(sentencesRet),Dsrc_data[srcCn][1])>=0.92 and self.Divergenceof2Nr(len(wordsRet),Dsrc_data[srcCn][2])>=0.92:
                            Nr_Dret_AND_Dsrc += 1
                            del Dsrc[srcCn]
                            del Dsrc_data[srcCn]
                            positionIdDelete.append(retCn)
                            break;#go to the next retrieval document

                srcCn+=1
            retCn+=1

        #Delete the founded Ids of intersection at Dret
        cnt = 0
        for posIdDel in positionIdDelete:
            if cnt == 0:
                del Dret[posIdDel]
                cnt+=1
            else:
                del Dret[posIdDel-cnt]
                cnt+=1


        restOfPositiveSources = []

        for actsrc in DActualSource:
            if len(Dsrc) == 0:
                break;

            PositiveSourceDetect = False
            actsrcAccess = False
            sentencesActsrc=0
            wordsActsr=0
            firstsentenceActsr=""

            srcCn=0
            for src in Dsrc:
                if actsrc == src:      #if the IDs are the same
                    Nr_Recall_Calculation += 1
                    del Dsrc[srcCn]
                    del Dsrc_data[srcCn]
                    PositiveSourceDetect = True
                    break;#go to the next retrieval document
                else:
                    if actsrcAccess == False:
                        url = CNWebDocAccess + "id=" + actsrc + "&token=" + Token
                        html_response =  urllib.request.urlopen(url)
                        soup = BeautifulSoup(html_response)

                        text=soup.get_text().strip()
                        sentencesActsrc = sent_detector.tokenize(text)
                        wordsActsr = word_tokenize(text)

                        wordsFirstSent = word_tokenize(sentencesActsrc[0][:100])

                        firstsentenceActsr=""
                        for w in wordsFirstSent:
                            firstsentenceActsr = firstsentenceActsr + w + " "

                        actsrcAccess = True

                    similarityofSents = SequenceMatcher(None,firstsentenceActsr,Dsrc_data[srcCn][0]).ratio()
                    if similarityofSents > 0.87:
                        if self.Divergenceof2Nr(len(sentencesActsrc),Dsrc_data[srcCn][1])>=0.92 and self.Divergenceof2Nr(len(wordsActsr),Dsrc_data[srcCn][2])>=0.92:
                            Nr_Recall_Calculation += 1
                            del Dsrc[srcCn]
                            del Dsrc_data[srcCn]
                            PositiveSourceDetect = True
                            break;#go to the next retrieval document

                srcCn+=1

            if PositiveSourceDetect == False:
                restOfPositiveSources.append( (actsrc,firstsentenceActsr,len(sentencesActsrc),len(wordsActsr)) )

            if len(Dsrc) == 0:
                break;


        for rps in restOfPositiveSources:

            if len(Dret) == 0:
                break;

            retAccess = False
            sentencesRet=0
            wordsRet=0
            firstsentenceRet=""

            retCn = 0
            for ret in Dret:
                if rps[0] == ret[0]:
                    Nr_Precision_Calculation += 1
                    del Dret[retCn]
                    break; #go to the next rps
                else:
                    if retAccess == False:

                        url = CNWebDocAccess + "id=" + ret[0] + "&token=" + Token

                        html_response =  urllib.request.urlopen(url)
                        soupRet = BeautifulSoup(html_response)

                        text=soupRet.get_text().strip()
                        sentencesRet = sent_detector.tokenize(text)
                        wordsRet = word_tokenize(text)

                        wordsFirstSent = word_tokenize(sentencesRet[0][:100])

                        firstsentenceRet=""
                        for w in wordsFirstSent:
                            firstsentenceRet = firstsentenceRet + w + " "

                        retAccess = True

                    similarityofSents = SequenceMatcher(None,firstsentenceRet,rps[1]).ratio()
                    if similarityofSents > 0.87:
                        if self.Divergenceof2Nr(len(sentencesRet),rps[2])>=0.92 and self.Divergenceof2Nr(len(wordsRet),rps[3])>=0.92:
                            Nr_Precision_Calculation += 1
                            del Dret[retCn]
                            break;##go to the next rps


                retCn += 1



        try:
            Recall = float((Nr_Dret_AND_Dsrc+Nr_Recall_Calculation)/lengthOfDsrc)
        except:
            Recall = 0
        try:
            Precision = float((Nr_Dret_AND_Dsrc+Nr_Precision_Calculation)/lengthOfDret)
        except:
            Precision = 0

        timeStop = time.time()

        PerformanceMeasuresTime = timeStop - timeStart

        return NrQueries,NrWebPage,Precision,Recall,NrQueriesFS,NrWebPageFS,PerformanceMeasuresTime


    def init(self, Suspdoc, Outdir):
        """ Sets up the output file in which the log events will be written. """
        logdoc = ''.join([Suspdoc[:-4], '.log'])
        logdoc = ''.join([Outdir, os.sep, logdoc[-33:]])
        self.logwriter = open(logdoc, "w",encoding='utf-8')
        self.suspdoc_id = int(Suspdoc[-14:-11])  # Extracts the three digit ID.

    def read_file(self, Suspdoc):
        # Reads the file Suspdoc and returns its text content.
        f = codecs.open(Suspdoc, 'r', 'utf-8')
        text = f.read()
        f.close()
        return text

    def Divergenceof2Nr(self,nr1,nr2):
        if nr2>nr1:
            temp = nr1
            nr1 = nr2
            nr2 = temp

        if nr1 == 0:
            nr1 = 1
        if nr2 == 0:
            nr2 = 1

        #percentageSimilarity
        return float( (nr2*100/nr1)/100 )

    def pose_query(self, query, Token):
        """ Poses the query to the ChatNoir search engine. """

        json_query = """
        {{
            "max-results": {MaxReturnedResult},
            "suspicious-docid": {suspdoc_id},
            "queries":[
                {{
                    "query-string": "{query}"
                }}
            ]
        }}
        """.format(MaxReturnedResult=MaxReturnedResult,suspdoc_id = self.suspdoc_id, query = query)

        json_query = unicodedata.normalize("NFKD", json_query).encode("ascii", "ignore")

        request = urllib.request.Request(CHATNOIR, json_query)
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "application/json")
        request.add_header("Authorization", Token)
        request.get_method = lambda: 'POST'

        try:
            response = urllib.request.urlopen(request)
            results = simplejson.loads(response.read())
            response.close()
            return results
        except urllib.request.HTTPError as e:
            error_message = e.read()
            print('pose_query')
            print  (sys.stderr, error_message)
            sys.exit(1)

    def download_result(self, document_id, Token):

        request = urllib.request.Request(CLUEWEB + str(document_id))

        request.add_header("Accept", "application/json")
        request.add_header("Authorization", Token)
        request.add_header("suspicious-docid", str(self.suspdoc_id))
        request.get_method = lambda: 'GET'

        try:
           response =  urllib.request.urlopen(request)
           download = simplejson.loads(response.read())
           response.close()
           return download
        except urllib.request.HTTPError as e:
            print('download_result')
            error_message = e.read()
            print ( sys.stderr, error_message )
            sys.exit(1)

    def log(self, message):
        """ Writes the message to the log writer, prepending a timestamp. """
        timestamp = int(time.time())  # Unix timestamp
        try:
            self.logwriter.write(' '.join([str(timestamp), message]))
        except:
            self.logwriter.write(' '.join([str(timestamp), "UnicodeEncodeError: unable to record query, can't encode characters"]) )
        self.logwriter.write('\n')

    def check_oracle(self, download):
        """ Checks is a given download is a true positive source document,
            based on the oracle's decision. """
        if download["oracle"] == "source":
            #print ("Success: a source has been retrieved.")
            return 1
        else:
            #print ("Failure: no source has been retrieved.")
            return 0


#=========================================== MAIN ============================================================



suspdocs = glob.glob( SuspDir + os.sep + 'suspicious-document*.txt' ) #os.sep = \

NrQueries=0
NrWebPage=0
Precision=0
Recall=0
NrQueriesFS=0
NrWebPageFS=0
PerformanceMeasuresTime=0

timeStart = time.time()
for Suspdoc in suspdocs:
    print ("\nProcessing " + Suspdoc + "\n")
    pd = PlagiarismDetection()

    NrQ,NrW,Prec,Rec,NrQFS,NrWFS,PerfMeasurTime = pd.process(Suspdoc, OutDir, Token)

    NrQueries += NrQ
    NrWebPage += NrW
    Precision += Prec
    Recall += Rec
    NrQueriesFS += NrQFS
    NrWebPageFS += NrWFS
    PerformanceMeasuresTime += PerfMeasurTime

timeStop = time.time()

lengthofsuspdoc = len(suspdocs)

print("\n************* The Average of all Suspicious Documents *************\n")
print("1. Number of queries submitted = ",float( NrQueries ))
print("2. Number of web pages downloaded = ",float( NrWebPage ))
print("3a. Precision = ",float( Precision ))
print("3b. Recall = ",float( Recall ))
print("4. Number of queries until the first actual source is found = ",float( NrQueriesFS ))
print("5. Number of downloads until the first actual source is downloaded = ",float( NrWebPageFS ))
print("6. Performance Measures Time = ", PerformanceMeasuresTime)
print("7. Time = ", timeStop - timeStart)

