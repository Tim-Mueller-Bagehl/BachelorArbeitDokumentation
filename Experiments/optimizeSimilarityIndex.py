from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
from InteractionManager import InteractionManager
import json
from datetime import datetime

@dataclass(frozen=True)
class Result:
    threshold: float
    finalScore: float   # maximize (0..1)
    incorrect: float    # minimize (0..1)

def pareto_front(results: List[Result]) -> List[Result]:
    """Nicht-dominierte Punkte: besser in finalScore UND nicht schlechter in incorrect."""
    front: List[Result] = []
    for r in results:
        dominated = False
        for o in results:
            if o is r:
                continue
            # o dominiert r, wenn o >= finalScore und o <= incorrect und mindestens eins strikt besser
            if (o.finalScore >= r.finalScore and o.incorrect <= r.incorrect and
                (o.finalScore > r.finalScore or o.incorrect < r.incorrect)):
                dominated = True
                break
        if not dominated:
            front.append(r)
    # sortiere schön: erst wenig incorrect, dann hoher finalScore
    front.sort(key=lambda x: (x.incorrect, -x.finalScore))
    return front

def optimize_similarity_index(
    evaluate: Callable[[float], Tuple[float, float]],
    data,
    system,
    thresholds: Optional[np.ndarray] = None,
    max_incorrect: Optional[float] = None,
    alpha: float = 0.5,
) -> Tuple[Result, List[Result], List[Result]]:
    """
    - evaluate(threshold) -> (finalScore, incorrect)
    - Wenn max_incorrect gesetzt: wähle bestes finalScore unter dieser Grenze (constrained).
    - Sonst: wähle per Utility = finalScore - alpha * incorrect (weighted).
    Gibt zurück: (best_result, pareto, all_results)
    """
    if thresholds is None:
        # Default: feines Raster
        thresholds = np.linspace(0.0, 1.0, 201)  # 201 Schritte

    all_results: List[Result] = []
    timetaken = []
    for t in thresholds:
        start = datetime.now()
        fs, inc = evaluate(system,data,float(t))
        fs = float(np.clip(fs, 0.0, 1.0))
        inc = float(np.clip(inc, 0.0, 1.0))
        all_results.append(Result(float(t), fs, inc))
        end = datetime.now()
        timetaken.append(end-start)

    pareto = pareto_front(all_results)

    if max_incorrect is not None:
        feasible = [r for r in all_results if r.incorrect <= max_incorrect]
        if feasible:
            best = max(feasible, key=lambda r: (r.finalScore, -r.incorrect))
        else:
            # falls nichts die Grenze einhält: nimm Pareto-Punkt mit minimal incorrect
            best = min(pareto, key=lambda r: (r.incorrect, -r.finalScore))
    else:
        # weighted utility: höher = besser
        best = max(all_results, key=lambda r: (r.finalScore - alpha * r.incorrect))

    return best, pareto, all_results,np.mean(timetaken)


# ----------------------------
# HIER DEINE LOGIK EINBAUEN
# ----------------------------
def evaluate(system,data,threshold: float) -> Tuple[float, float]:
    """
    TODO: Hier führst du deine Experimente/Retrieval aus und berechnest:
    - finalScore: Anteil korrekter Antworten (0..1) => maximieren
    - incorrect: Anteil fälschlich rausgesuchter Fakten (0..1) => minimieren
    """
    runningscore = []
    runningIncorrect = []
    for eintrag in  data["eintraege"]:
        preparedInput = system.apiCommunicationSystem.prepareInput(eintrag["frage"])
        retrivedText, _ = system.directoryManagementSystem.searchMemoryDirectory("1",preparedInput,simmilarityIndex= threshold,normalize=True)

        if retrivedText == []:
            runningscore.append(0)
            runningIncorrect.append(0)
            continue
        correct = sum(1 for x in eintrag["antworten"] if x in retrivedText)
        incorrect = sum(1 for x in retrivedText if x not in eintrag["antworten"])
        runningscore.append(correct/len(retrivedText))
        runningIncorrect.append(incorrect/len(retrivedText))
    finalscore = sum(runningscore)/len(runningscore) if runningscore != [] else 0
    incorrect = sum(runningIncorrect)/len(runningIncorrect) if runningIncorrect != [] else 0
    #print(f"finalscore:{finalscore*100:.2f}%")
    #print(f"incorrect:{incorrect*100:.2f}%")
    return finalscore,incorrect





def getJson(Json:str):
    with open(Json, 'r') as f:
        data = json.load(f)

    return data
if __name__ == "__main__":
    embeddingalgorithms =["multi-qa-mpnet-base-dot-v1","bi-encoder_msmarco_bert-base_german","paraphrase-multilingual-mpnet-base-v2","multi-qa-MiniLM-L6-dot-v1"]
    #embeddingalgorithms = ["bi-encoder_msmarco_bert-base_german"]
    for algorithm in embeddingalgorithms:
        system = InteractionManager(general_embeddingstrategy=algorithm)
        system.VectorDatabase.deleteDirectory("1")
        system.VectorDatabase.createDirectory("1")
        data = getJson("Experiments/Experiment1v3.json")
        alle_angaben = data["alle_antworten"]
        embeddings = system.ModelGateway.generalEmbeddingstrategy(alle_angaben)
        system.VectorDatabase.addNewVectorsToDirectory("1",embeddings,alle_angaben,True)

        best, pareto, all_res, timetaken = optimize_similarity_index(
            evaluate,
            data,system,
            thresholds=np.linspace(0.0, 0.95, 200),
            max_incorrect=0.5,   # z.B. höchstens 10% falsche Fakten erlaubt
            alpha=0.5             # nur relevant, wenn max_incorrect=None
            )
        print("BEST:", best,"for:",algorithm)

    