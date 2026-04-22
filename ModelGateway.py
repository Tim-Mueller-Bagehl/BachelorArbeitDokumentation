from datetime import datetime
import os
import time
from openai import OpenAI
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from VectorDatabase import VectorDatabaseSystem
 
class ModelGateway:

    def __init__(self,
                 directoryManangementSystem :VectorDatabaseSystem,
                 apiKey : str | None,
                 model : str = "gpt-4o-mini",
                 temperature : float = 0.7,
                 similarityIndex : float = 0.85,
                 prompt_for_factextraction : str = None,
                 prompt_for_general_APIcall_retrivedFacts : str = None,
                 prompt_for_general_APIcall_noretrivedFacts : str = None,
                 prompt_to_update_general_Userinformation : str = None,
                 embeddingstrategy : str = "multi-qa-MiniLM-L6-dot-v1"
                 ):
        self.client = OpenAI(api_key=apiKey) if apiKey is not None else OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.similarityIndex = similarityIndex
        self.prompt_for_factextraction = prompt_for_factextraction if prompt_for_factextraction is not None else "Du bist ein Spion der eine Unterhaltung zwischen einer Person und einem Chatbot abgefangen hat.Deine Aufgabe ist es Fakten zu der Person anhand der letzten Frage des Chatbots und der Antwort der Person zu sammeln. Diese Fakten sollen später von einem anderen chatbot genutzt werden.Trenne alle Fakten jeweils mit einem | . Bündel dabei Fakten zusammen Nutze nur die Antwort des Users und nicht die Frage"
        self.prompt_for_general_APIcall_retrivedFacts = prompt_for_general_APIcall_retrivedFacts if prompt_for_general_APIcall_retrivedFacts is not None else "Du bist ein freundlicher Mitarbeiter in einem Altersheim. Deine Aufgabe ist es eine Menschliche Unterhaltung mit deinem Gegenüber zu führen und falls angemessen den beigefügten Kontext für deine Antwort nutzen"
        self.prompt_for_general_APIcall_noretrivedFacts = prompt_for_general_APIcall_noretrivedFacts if prompt_for_general_APIcall_noretrivedFacts is not None else "Du bist ein freundlicher Mitarbeiter in einem Altersheim. Deine Aufgabe ist es eine Menschliche Unterhaltung mit deinem Gegenüber zu führen"
        self.prompt_to_update_general_Userinformation = prompt_to_update_general_Userinformation if prompt_to_update_general_Userinformation is not None else "Du erhälst ein allgemeingültiges Benutzerprofil zu einem User in Kombination mit einer Anfrage, die dieser Benutzer gestellt hat. Nutze diese Anfrage, um das Benutzerprofil fall nötig zu updaten. Informationen wie Name und Alter sollten immer im Benutzerprofil bleiben. Das Benutzerprofil sollte nicht länger als 150 Worte lang sein. Vermeide wiedersprüche, gib nur die Geupdateten Generellen Informationen wieder. Wenn du keine generellen Informationen bekommen hast nutze den Query um generelle Informationen anzulegen." 
        self.directoryManagementSystem = directoryManangementSystem
        match(embeddingstrategy):
            case "multi-qa-MiniLM-L6-dot-v1":
                self.embeddingStrategy = SentenceTransformer("multi-qa-MiniLM-L6-dot-v1")#384
            case "paraphrase-multilingual-mpnet-base-v2":
                self.embeddingStrategy = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")#768
            case "bi-encoder_msmarco_bert-base_german":
                self.embeddingStrategy = SentenceTransformer("PM-AI/bi-encoder_msmarco_bert-base_german")#768
            case "multi-qa-mpnet-base-dot-v1":
                self.embeddingStrategy = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")#768
            case _:
                self.embeddingStrategy = SentenceTransformer(embeddingstrategy)
            

    def processInput(self,ID : str,query : str, shortTermMemory : list[str]):
        """Main Function to call. Processes the request, searches for relevant facts and handels the request to the API

        Args:
            ID (str): ID of the User
            query (str): The request

        Returns:
            _type_: response of the LLM
        """
        answer = self.prepareInputForAPIRequest(ID,query,shortTermMemory)
        if shortTermMemory == []:
            self.retriveAndSaveFacts(ID,query,"")
        else:
            self.retriveAndSaveFacts(ID,query,shortTermMemory[-1])
        return answer 



    def callAPI(self,query : str, shortTermMemory : str ,generalInformation : str ,retrivedText = None):
        """Performs the actual API request using the retrived text and the query of the Person

        Args:
            query (str): Input
            retrivedText (_type_, optional): Relevant facts. Defaults to None.

        Returns:
            _type_: API response
        """
        if retrivedText == None:
                messages = [{"role":"system","content":self.prompt_for_general_APIcall_noretrivedFacts},
                    {"role":"user","content":f"message:{query}"}]
        else:
            messages = [{"role":"system","content":self.prompt_for_general_APIcall_retrivedFacts},
                        {"role":"user","content":f"Previous Conversation:{shortTermMemory} \n general Information: {generalInformation} \n  Content: {retrivedText}\n\n Question:{query}"}]
            
        response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                temperature = self.temperature
        )
        return response.choices[0].message.content
        
    def gatherFactsFromInput(self,query : str, question : str):
        """Takes the Input and question of the AI and gathers facts about the Person

        Args:
            query (str): Input
            question (str): question

        Returns:
            _type_: facts
        """
        messages = [{"role":"system","content":self.prompt_for_factextraction},
                {"role":"user","content":f"Frage:{question}\n\n Antwort:{query}"}]
        response = self.client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = self.temperature
        )
        facts = response.choices[0].message.content.split("|")
        facts = [a.strip().lower() for a in facts if a != "" or a != '']
        self.writeExtractedFacts(facts)
        return facts
    
    def updateGeneralInformation(self, ID : str, generalInformation : str, query : str):
        """Upadates the User-Profile using the last User-Query

        Args:
            ID (str): User-ID
            generalInformation (str): User-Profile
            query (str): last User Query

        Returns:
            _type_: updated Profile
        """
        messages = [{"role":"system","content":self.prompt_to_update_general_Userinformation},
                {"role":"user","content":f"General Information: {generalInformation}\n\n Query:{query}"}]
        response = self.client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = self.temperature
        )
        generalInformation = response.choices[0].message.content
        self.directoryManagementSystem.updateGeneralInformation(ID,generalInformation)
        return generalInformation

    def generalEmbeddingstrategy(self,sentences : list[str]):
        """Returns a list of Embeddings where each Embedding represents one Sentence

        Args:
            facts (list[str]): List of Sentences

        Returns:
            _type_: A Numpyarray containing all the Embeddings
        """
        #Removes .!? and whitespaces at the beginning and End. Also casts it to lower and removes all empty strings
        if type(sentences) is str and (sentences != "" or ''):
            sentences = (sentences.translate(str.maketrans("","",".!?"))).strip().lower()
        else:
            sentences = [(a.translate(str.maketrans("","",".!?"))).strip().lower() for a in sentences if a != "" or a != '']
        embeddings = self.embeddingStrategy.encode(sentences,convert_to_numpy=True,normalize_embeddings=True)
        return embeddings



    def prepareInput(self,query:list[str]):
        """Creates one "fast" Embedding to search the Faiss Index and one "precise" Embedding to rerank the top 5 Returned Embeddings

        Args:
            query (list[str]): The Input String

        Returns:
            _type_: fastQuery,preciseQuery
        """
        input =[]
        if type(query) is str:
            input.append(f"Welche Fakten könnten für dieses Input {query} relevant sein?")
        else:
            for sentence in query:
                input.append(f"Welche Fakten könnten für dieses Input {sentence} relevant sein?")
        embeddings = self.generalEmbeddingstrategy(input)
        if embeddings.ndim == 1: 
            embeddings = embeddings.reshape(1,-1)          
        return embeddings
       

    def retriveAndSaveFacts(self,ID : str, query : str, question : str = ""):
        """Takes the Input Query and filters new Information about the Person using the Input and the previous response of the AI

        Args:
            ID (str): ID of the person
            query (str): The Input of the Person
            question (str): Question of the AI
        """
        facts = self.gatherFactsFromInput(query,question)
        embeddings = self.generalEmbeddingstrategy(facts)
        self.directoryManagementSystem.addNewVectorsToDirectory(ID,embeddings,facts)

    def prepareInputForAPIRequest(self,ID : str, query : str,shortTermMemory:list[str]):
        """Splits Input -> Turns it into Embeddings -> searches Faiss Index -> Reranks outputs ->calls API ->returns response

        Args:
            ID (str): ID of the person
            query (str): Input of the Person

        Returns:
            _type_: response of the API
        """
        sentences = re.split(r'[.?!]',query)
        embeddings = self.prepareInput(sentences)
        retrivedText, generalInformation = self.directoryManagementSystem.searchMemoryDirectory(ID,embeddings,simmilarityIndex=self.similarityIndex)
        self.writeRetrivedFacts(retrivedText)
        retrivedText = "\n\n".join(retrivedText)
        shortTermMemory = "\n".join(shortTermMemory)
        self.updateGeneralInformation(ID,generalInformation,query)
        answer = self.callAPI(query,shortTermMemory,generalInformation,retrivedText )
        return answer

    ######################################################################################
    #Fuctions Used for Experiments. They use the System Fuctions but they had to be adapted to the Experiment Enviorment
    def prepareInputExperiment3(self,ID : str, query : str, validAnswers : list[str]):
        sentences = re.split(r'[.?!]',query)
        
        embeddings = self.prepareInput(sentences)
        retrivedText, generalInformation = self.directoryManagementSystem.searchMemoryDirectory(ID,embeddings,simmilarityIndex=self.similarityIndex, k = 5)
        self.writeRetrivedFacts(retrivedText)
        retrivedText = "\n\n".join(retrivedText)

        validAnswers = "\n\n".join(validAnswers)
        answer = self.callAPIExperiment3(ID,query,retrivedText,validAnswers,generalInformation)
        return answer
    
    def callAPIExperiment3(self,ID : str, query : str, retrivedText : str, validAnswers : str,generalInformation):
        if retrivedText == None:
            messages = [{"role":"system","content":self.prompt_for_general_APIcall_noretrivedFacts},
                {"role":"user","content":f" message:{query}\n\n Valid answers: {validAnswers}"}]
        else:
            messages = [{"role":"system","content":self.prompt_for_general_APIcall_retrivedFacts},
                        {"role":"user","content":f"Content: {retrivedText}\n\n userProfile:{generalInformation} \n\nQuestion:{query} \n\n Valid answers: {validAnswers}"}]
            
        response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                temperature = self.temperature
        )
        return response.choices[0].message.content
    def testFactretrival(self,ID,query):
        """Function to test factretrival. Should be deleted in the final product
        @TODO delete later

        Args:
            ID (_type_): _description_
            query (_type_): _description_

        Returns:
            _type_: _description_
        """
        sentences = re.split(r'[.?!]',query)
        embeddings = self.prepareInput(sentences)
        retrivedText = self.directoryManagementSystem.searchMemoryDirectory(ID,embeddings,simmilarityIndex=self.similarityIndex)
        return retrivedText
    
    def testFactextraction(self,query,expectedFacts,threshhold):
        """Function to test factExtraction using a query and expected facts that compares the result of the query to the expected facts using the chosen Wordembedding strategy

        Args:
            query (_type_): Input string
            expectedFacts (_type_): list of expected facts for that string
            threshhold (_type_): Threshhold for the cosine similarity

        Returns:
            _type_: List of similar facts
        """
        facts = self.gatherFactsFromInput(query,"")
        factEmbedding = self.generalEmbeddingstrategy(facts)
        expectedFactEmbeddings = self.generalEmbeddingstrategy(expectedFacts) 
        dotMatrix = factEmbedding @ expectedFactEmbeddings.T
        indices = np.argwhere(dotMatrix > threshhold)
        return list(set([[facts[i],expectedFacts[j]] for i,j in indices]))

    ########################################################################################################################################################

    def writeRetrivedFacts(self,text : list[str]):
        with open("protocol.txt","a",encoding="utf-8") as f:
            now = time.time()
            f.write("\nRetrivedFacts:\n")
            for index,fact in enumerate(text):
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d %H:%M:%S")
                f.write(f"\nFact{index+1}:{fact}        {timestamp}")

    def writeExtractedFacts(self,text : list[str]):
        with open("protocol.txt","a",encoding="utf-8") as f:
            now = time.time()
            f.write("\nExtractedFacts:\n")
            for index,fact in enumerate(text):
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d %H:%M:%S")
                f.write(f"\nFact{index+1}:{fact}        {timestamp}")
