from datasets import load_dataset
import json
import pandas as pd
import hashlib


ds = load_dataset("YuWangX/Memalpha", split="train")#https://arxiv.org/html/2402.16288v1
per = ds.filter(lambda x: x.get("data_source") == "perltqa")

print(per.column_names)
ex0 = per[0]
print({k: type(ex0[k]).__name__ for k in ex0})

# 👉 Du musst hier einmal schauen, welche Spalten bei dir "memories" und "qas" sind.
# In vielen Memalpha-Subsets sind das Strings, die JSON enthalten:
# - eine Spalte mit einer JSON-Liste von Memory-Chunks
# - eine Spalte mit einer JSON-Liste von {"question":..., "answer":...}
MEM_COL = "chunks"    # <- ggf. anpassen nach print()
QA_COL  = "questions_and_answers"   # <- ggf. anpassen nach print()

docs = []
queries = []

for i, row in enumerate(per):
    memories = json.loads(row[MEM_COL])  # list[str] oder list[...]
    qas = json.loads(row[QA_COL])        # list[dict]

    person_id = row.get("instance_id", f"p{i}")

    # Documents: jede Memory-Zeile als eigenes Doc
    doc_ids = []
    for j, mem in enumerate(memories):
        text = mem if isinstance(mem, str) else json.dumps(mem, ensure_ascii=False)
        doc_id = hashlib.sha1(f"{person_id}|{j}|{text}".encode("utf-8")).hexdigest()
        doc_ids.append(doc_id)
        docs.append({"doc_id": doc_id, "person_id": person_id, "text": text})

    # Queries: ohne offizielles "reference memory" hast du meist KEIN eindeutiges positive_doc_id.
    # -> Speichere erstmal query+answer+person_id, und mappe positives später (heuristisch oder per Label falls vorhanden).
    for k, qa in enumerate(qas):
        queries.append({
            "query_id": f"{person_id}_q{k}",
            "person_id": person_id,
            "query": qa.get("question"),
            "answer": qa.get("answer"),
        })

pd.DataFrame(docs).to_json("documents.jsonl", orient="records", lines=True, force_ascii=False)
pd.DataFrame(queries).to_json("queries.jsonl", orient="records", lines=True, force_ascii=False)
print("Wrote documents.jsonl + queries.jsonl")
