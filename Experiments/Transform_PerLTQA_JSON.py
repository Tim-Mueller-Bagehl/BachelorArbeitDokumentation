import json
import re
from pathlib import Path

SUMMARY_MARK = "Summary:"
CONTENT_MARK = "Content:"
DIALOG_HDR_RE = re.compile(r"The following are the dialogues\.?", re.IGNORECASE)
DIALOG_DATE_RE = re.compile(r"(?im)^\s*Dialogue happened at[^\n]*\s*$")
SPEAKER_LINE_RE = re.compile(r"^\s*<([^>]+)>\s*(.*)\s*$")  # <AI assistant> bla

def cut_to_summary(raw: str) -> str:
    i = raw.find(SUMMARY_MARK)
    return raw[i:] if i != -1 else raw

def slice_between(text: str, start: int, start_len: int, end: int) -> str:
    if start == -1:
        return ""
    a = start + start_len
    b = end if end != -1 else len(text)
    return text[a:b].strip()

def parse_dialog_block_to_turns(block: str) -> list[str]:
    """
    Wandelt einen Dialogblock in eine Liste von Turns um:
    [assistant_utterance, user_utterance, assistant_utterance, user_utterance, ...]
    Entfernt Datumszeilen, entfernt <Speaker>-Tags, normalisiert Fortsetzungszeilen.
    """
    turns: list[str] = []

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if DIALOG_DATE_RE.match(line):
            continue

        m = SPEAKER_LINE_RE.match(line)
        if m:
            utter = m.group(2).strip()
            if utter:
                turns.append(utter)
        else:
            # Falls eine Zeile ohne <...> kommt, hänge sie an den letzten Turn an
            if turns:
                turns[-1] = (turns[-1] + " " + line).strip()

    # Optional: Wenn ein Turn leer ist (sollte nicht vorkommen), rausfiltern
    turns = [t for t in turns if t]
    return turns

def parse_sections(raw_text: str) -> dict:
    text = cut_to_summary(raw_text).strip()

    s_idx = text.find(SUMMARY_MARK)
    c_idx = text.find(CONTENT_MARK)
    d_match = DIALOG_HDR_RE.search(text)
    d_idx = d_match.start() if d_match else -1

    summary_end = min([x for x in [c_idx, d_idx] if x != -1], default=-1)
    summary = slice_between(text, s_idx, len(SUMMARY_MARK), summary_end)

    content_end = d_idx
    content = slice_between(text, c_idx, len(CONTENT_MARK), content_end)

    dialogs = []
    if d_match:
        dialog_text = text[d_match.end():].strip()

        # Split an Datums-Trennern "Dialogue happened at ..."
        # Wir machen daraus "Dialog-Blöcke".
        # Trick: Wir splitten an Zeilen, die mit "Dialogue happened at" anfangen.
        blocks = re.split(r"(?im)^\s*Dialogue happened at[^\n]*\n", dialog_text)

        # Falls es kein führendes Datum gibt, kann block[0] schon Inhalt haben – ok.
        for b in blocks:
            b = b.strip()
            if not b:
                continue
            turns = parse_dialog_block_to_turns(b)
            if turns:
                dialogs.append(turns)

    return {"summary": summary, "content": content, "dialogs": dialogs}

def transform_jsonl(in_path: str, out_path: str):
    in_path = Path(in_path)
    out_path = Path(out_path)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)

            raw_text = row.get("text", "")
            sections = parse_sections(raw_text)

            # text komplett entfernen und durch summary/content/dialogs ersetzen
            row.pop("text", None)
            row["summary"] = sections["summary"]
            row["content"] = sections["content"]
            row["dialogs"] = sections["dialogs"]

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    transform_jsonl("documents.jsonl", "documents_structured.jsonl")
    print("Fertig: documents_structured.jsonl")