from InteractionManager import InteractionManager
import json
import numpy as np
from sentence_transformers import SentenceTransformer





def Experiment1(system : InteractionManager, path : str, embeddingstrategy : str,zeromatchProtocol : str):
    with open("Protocol/Experiment1.txt","a",encoding="utf-8") as f:
        f.write(f"\n{embeddingstrategy}\n\n")
    data = getJson(path)
    alle_angaben = data["alle_antworten"]
    embeddings = system.ModelGateway.generalEmbeddingstrategy(alle_angaben)
    system.VectorDatabase.addNewVectorsToDirectory("1",embeddings,alle_angaben,True)
    runningscore = []
    runningIncorrect = []
    zeromatchescounter = 0
    for eintrag in  data["eintraege"]:
        preparedInput = system.ModelGateway.prepareInput(eintrag["frage"])
        retrivedText,_ = system.VectorDatabase.searchMemoryDirectory("1",preparedInput,simmilarityIndex= system.ModelGateway.similarityIndex,normalize=True)
        if retrivedText == []:
            runningscore.append(0)
            continue
        correct = sum(1 for x in retrivedText if x in eintrag["antworten"])
        incorrect = sum(1 for x in retrivedText if x not in eintrag["antworten"])
        if correct == 0:
            zeromatchescounter+=1
            writeZeroMatches(eintrag,retrivedText,zeromatchProtocol)
        runningscore.append(correct/len(retrivedText))
        runningIncorrect.append(incorrect/len(retrivedText))
        writeProtocolForExperiment1(eintrag,retrivedText)
    finalscore = sum(runningscore)/len(runningscore)
    incorrect = sum(runningIncorrect)/len(runningIncorrect)
    with open("Protocol/Experiment1.txt","a",encoding="utf-8") as f:
        f.write(f"finalscore:{finalscore*100:.2f}%\n")
        f.write(f"incorrect:{incorrect*100:.2f}%\n")
    print(f"finalscore:{finalscore*100:.2f}%")
    print(f"incorrect:{incorrect*100:.2f}%")
    print(f"zeromatches:{zeromatchescounter}")
    

def writeZeroMatches(input,facts,path):
    with open(path,"a",encoding="utf-8") as f:
        frage = input["frage"]
        f.write(f"\n\nQuestion:{frage}\nExpected facts:\n")
        antworten = input["antworten"]
        for line in antworten:
            f.write(f"{line}\n")
        f.write("\nWrong facts:\n")
        for line in facts:
            f.write(f"{line}\n")


                
def writeProtocolForExperiment1(input,facts):
    with open("Protocol/Experiment1.txt","a",encoding="utf-8") as f:
        frage = input["frage"]
        f.write(f"Question:{frage}\nFacts:\n")
        input = input["antworten"]
        for line in input:
            if line in facts:
                f.write(f"{line} ✅\n")
            #else:
                #f.write(f"{line} ❌\n")
        f.write("\nWrong facts:\n")
        for fact in facts:
            if fact not in input:
                f.write(f"{fact}\n")
        f.write("\n")




def getJson(Json:str):
    with open(Json, 'r') as f:
        data = json.load(f)

    return data

def getJsonLines(path : str):
    rows = []
    with open(path,"r", encoding= "utf-8") as f:
        for line_no,line in enumerate(f,start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj,str):
                    obj = json.loads(obj)
                rows.append(obj)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on Line {line_no} {e}\n Line: {line[:200]!r}")
    
    return rows

def runExperiment1():
    open("Protocol/Experiment1.txt","w").close()
    embeddingalgorithms = ["multi-qa-mpnet-base-dot-v1","bi-encoder_msmarco_bert-base_german","paraphrase-multilingual-mpnet-base-v2","multi-qa-MiniLM-L6-dot-v1"]
    for algorithm in embeddingalgorithms:
        last_line_number = 0
        with open("Protocol/Experiment1.txt", "r", encoding="utf-8") as f:
            for last_line_number, _ in enumerate(f, start=1):
                pass
        zeromatchProtocolPath = f"Protocol/Experiment1{algorithm}.txt"
        print(algorithm)
        system = InteractionManager(general_embeddingstrategy=algorithm)
        system.VectorDatabase.deleteDirectory("1")
        system.VectorDatabase.createDirectory("1")
        path = "Experiments/Experiment1v4.json"
        Experiment1(system,path,algorithm,zeromatchProtocolPath)








def Experiment2(system :InteractionManager,path:str,threshhold):
    data = getJson(path)
    runningscore = []
    zeromatchLog = []   
    for eintrag in data:
        id = eintrag["id"]
        aussage = eintrag["aussage"]
        fakten = eintrag["fakten"]
        
        apiResponse = system.ModelGateway.gatherFactsFromInput(aussage,"")
        responseEmbeddings = system.ModelGateway.generalEmbeddingstrategy(apiResponse)
        faktenEmbeddings = system.ModelGateway.generalEmbeddingstrategy(fakten)
        dotMatrix = responseEmbeddings @ faktenEmbeddings.T
        
        top1idx = dotMatrix.argmax(axis=1)
        top1val = dotMatrix.max(axis=1)
        mask = top1val >= threshhold
        badmatchesMask = top1val < threshhold
        pairs_all = np.column_stack((np.arange(dotMatrix.shape[0]), top1idx))
        indices = pairs_all[mask]
        badindices = pairs_all[badmatchesMask]
        matchingFacts = list(set([(apiResponse[i],fakten[j]) for i,j in indices]))
        badMatches = list(set([(apiResponse[i],fakten[j]) for i,j in badindices]))
        correct = len(matchingFacts)/len(apiResponse) if len(apiResponse) != 0  else 0# Number of correct facts out of all the given options
        runningscore.append(correct)
        writeProtocolForExperiment2(aussage,matchingFacts,badMatches)
        if(correct == 0):
            zeromatchLog.append([aussage,apiResponse,fakten])
        if badMatches != []:
            print(badMatches)

    print(f"FinalScore:{((sum(runningscore)/len(runningscore))*100):.2f}%")
    print(f"ZeromatchScore:{(len(zeromatchLog)/len(runningscore)):.2f}%")
    if(zeromatchLog != []):
        writeZeromatchesExperiment2(zeromatchLog)
    with open("Protocol/Experiment2.txt","a",encoding="utf-8") as f:
        f.write(f"FinalScore:{((sum(runningscore)/len(runningscore))*100):.2f}%")
        f.write(f"ZeromatchScore:{(len(zeromatchLog)/len(runningscore)):.2f}%")


def writeProtocolForExperiment2(aussage,matchingfacts,badmatches):
    with open("Protocol/Experiment2.txt","a",encoding="utf-8") as f:
        f.write(f"Query:{aussage}\nGood Matches:\n")
        for match in matchingfacts:
            f.write(f"{match}\n")
        if badmatches != []:
            f.write("\nBad Matches: \n")
        for match in badmatches:
            f.write(f"{match}\n")
        f.write("\n")

def writeZeromatchesExperiment2(zeromatches):
    with open("Protocol/Experiment2.txt","a",encoding="utf-8") as f:
        f.write("\n Zeromatches:\n\n")
        for zeromatch in zeromatches:
            f.write(f"Query:{zeromatch[0]}\n")
            f.write("AI responses:\n")
            for response in zeromatches[1]:
                f.write(f"{response}\n")
            f.write("Fakten:\n")
            for fakt in zeromatches[2]:
                f.write(f"{fakt}\n")



    


def runExperiment2():
    open("Protocol/Experiment2.txt","w").close()
    system = InteractionManager()
    system.VectorDatabase.deleteDirectory("1")
    system.VectorDatabase.createDirectory("1")
    path = "Experiments/Experiment2.json"
    threshhold = 0.9213567839195979
    Experiment2(system,path,threshhold)

def Experiment3(system : InteractionManager, path : str,maxpeople : int, skipuntill : int):
    data = getJsonLines(path)
    knownPeople = []
    for person in data:
        ID = str(person["person_id"])
        
        generalInformation = ""
        if len(knownPeople) > maxpeople: break
        if ID not in knownPeople:
            system.VectorDatabase.deleteDirectory(ID)
            system.VectorDatabase.createDirectory(ID)
            knownPeople.append(ID)
            print(f"starting with person:{ID}")
            generalInformation = ""
        if int(ID) < skipuntill: continue
        system.ModelGateway.retriveAndSaveFacts(ID,person["content"])
        for dialog in person["dialogs"]:
            for p in range(0,len(dialog),2):
                try:
                    system.ModelGateway.retriveAndSaveFacts(ID,dialog[p+1],dialog[p])
                    generalInformation = system.ModelGateway.updateGeneralInformation(ID,generalInformation,dialog[p])
                except IndexError:
                    system.ModelGateway.retriveAndSaveFacts(ID,dialog[p])
                    generalInformation = system.ModelGateway.updateGeneralInformation(ID,generalInformation,dialog[p])

def Experiment3part2(system : InteractionManager,path : str, threshhold : float, maxpeople :int):
    data = getJsonLines(path)
    correct = []
    knownpeople = []
    perProfileScore= []
    validAnswers = []
    runningscore = 0
    i = 0
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")#semantic simmilarity Embedding to compare the two answers original paper:https://proceedings.neurips.cc/paper/2020/file/c3a690be93aa602ee2dc0ccab5b7b67e-Paper.pdf Then you add a footnote of the model you are using --> "sentence-transformers/all-mpnet-base-v2"
    for entry in data:
        if i%100 == 0:
            validAnswers = []
            if knownpeople != []:
                perProfileScore.append([knownpeople[:-1],(runningscore/100)*100])
                runningscore = 0
            for p in range(100):
                validAnswers.append(data[i+p]["answer"])
        i+=1
        ID = str(entry["person_id"])
        if len(knownpeople) > maxpeople:break
        if ID not in knownpeople:
            print(f"Starting on Person:{ID}")
            knownpeople.append(ID)

        question = entry["query"]
        answer = entry["answer"]
        response = system.ModelGateway.prepareInputExperiment3(ID,question,validAnswers)
        embeddings = model.encode([answer,response],normalize_embeddings=True)
        
        cosineSimmilarity = embeddings[0] @ embeddings[1]
        if cosineSimmilarity > threshhold:
            correct.append(1)
            runningscore += 1
            writeProtocolExperiment3Part2(question,answer,response,True)
        else:
            correct.append(0)
            writeProtocolExperiment3Part2(question,answer,response,False)
    
    score = sum(correct)/len(correct)
    print(f"Final Score:{(score*100):.2f}%")

def writeProtocolExperiment3Part2(question,answer,response,correct):
    with open("Protocol/Experiment3.txt","a",encoding="utf-8") as f:
        if correct:
            f.write(f"Question:{question} ✅\n")
        else:
            f.write(f"Question:{question} ❌\n")
        f.write(f"Correct Answer:{answer}\n")
        f.write(f"Given Response{response}\n\n")


def runExperiment3(maxusers: int,skip : int):
    modells = ["bi-encoder_msmarco_bert-base_german"]
    for model in modells: 
        factextractionPrompt = "You are a spy that intercepted the conversation between a person and a Chatbot. It is your job to gather facts about that Person based on the last question of the chatbot and the Persons answer. Those fact will be used by another chatbot at a later date. Always seperate facts with a |. Bundle facts if possible and only use the Response of the User for facts and not the question."
        promptForGeneralAPICall = "It is your job to use the Information provided to you to choose the correct answer out of the valid answers Provided. Just responde with the correct answer no other text required"
        promptToUpdateGeneralUserinformation = "You get general Information about a user in Combination with a message sent by that user. Update the General Information, if necessary to add new Information. The Information should be as general as possible. The Information will be used at a later point to personalise a conversation with a LLM. General Information should be no longer than 150 Words, avoid contradictions, only return the updated general Informations, If there are no General Informations provided use the Query to start a new set of general Informations "
        embeddingstrategy = "multi-qa-mpnet-base-dot-v1"
        system = InteractionManager(prompt_to_update_general_Userinformation=promptToUpdateGeneralUserinformation,general_embeddingstrategy=model,prompt_for_factextraction=factextractionPrompt,prompt_for_general_APIcall_retrivedFacts=promptForGeneralAPICall,prompt_for_general_APIcall_noretrivedFacts=promptForGeneralAPICall)
        path = "documents_structured.jsonl"
        #system.directoryManagementSystem.shutDownMemorySystem()
        Experiment3(system,path,maxusers,skip)
        runExperiment32(maxusers,system)

def runExperiment3part2(maxusers:int):
    open("Protocol/Experiment3.txt","w").close()
    model =  "multi-qa-mpnet-base-dot-v1"
    factextractionPrompt = "You are a spy that intercepted the conversation between a person and a Chatbot. It is your job to gather facts about that Person based on the last question of the chatbot and the Persons answer. Those fact will be used by another chatbot at a later date. Always seperate facts with a |. Bundle facts if possible and only use the Response of the User for facts and not the question."
    promptForGeneralAPICall = "It is your job to use the Information provided to you to choose the correct answer out of the valid answers Provided. Just responde with the correct answer no other text required"
    promptToUpdateGeneralUserinformation = "You get general Information about a user in Combination with a message sent by that user. Update the General Information, if necessary to add new Information. The Information should be as general as possible. The Information will be used at a later point to personalise a conversation with a LLM. General Information should be no longer than 150 Words, avoid contradictions, only return the updated general Informations, If there are no General Informations provided use the Query to start a new set of general Informations "
    system = InteractionManager(prompt_to_update_general_Userinformation=promptToUpdateGeneralUserinformation,general_embeddingstrategy=model,prompt_for_factextraction=factextractionPrompt,prompt_for_general_APIcall_retrivedFacts=promptForGeneralAPICall,prompt_for_general_APIcall_noretrivedFacts=promptForGeneralAPICall)
    path = "queries.jsonl"
    threshhold = 0.8
    Experiment3part2(system,path,threshhold,maxusers)

def runExperiment32(maxusers : int,system : InteractionManager):
    open("Protocol/Experiment3.txt","w").close()
    path = "queries.jsonl"
    threshhold = 0.9
    Experiment3part2(system,path,threshhold,maxusers)

#maxusers = 27
#skip = 504
#runExperiment3(maxusers,skip)
#runExperiment3part2(10)
#runExperiment3(10,0)



def checkExperimentDataset():
    data = getJson("Experiments/Experiment1v4.json")
    i = []
    checked = False
    print(len(data["eintraege"]))
    for eintrag in data["eintraege"]:
        if not checked and len(eintrag["antworten"]) == 42:
            print(eintrag["antworten"])
            checked = True
        #print(eintrag["antworten"])
        i.append(len(eintrag["antworten"]))
    
    print(np.mean(i))
    print(set(i))

checkExperimentDataset()
#runExperiment1()