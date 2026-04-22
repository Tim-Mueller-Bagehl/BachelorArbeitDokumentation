import os
from pydub import AudioSegment
from tqdm import tqdm
from pathlib import Path
import random
from typing import List, Tuple, Callable, Dict, Any
import shutil
from SpeakerIdentification import SpeakerIdentificationSystem
input_root = Path(__file__).parent /"SpeakerRecognitionTests"/"LibriSpeech"
output_root = Path(__file__).parent / "SpeakerRecognitionTests"/"LibriSpeech_wav"
split_root = Path(__file__).parent / "SpeakerRecognitionTests"/"LibriSpeech_split"

input_root1 = Path(__file__).parent /"SpeakerRecognitionTests"/"LibriSpeech1"
output_root1 = Path(__file__).parent / "SpeakerRecognitionTests"/"LibriSpeech1_wav"
split_root1 = Path(__file__).parent / "SpeakerRecognitionTests"/"LibriSpeech1_split"


Sample = Tuple[str, str]


def parse_speaker_id_from_folder(folder_name: str) -> str:
    """
    Erwartet Ordnernamen im Format 'Speaker_00xx'.
    Gibt die letzten zwei Ziffern 'xx' als Personen-ID zurück.
    """
    suffix = folder_name.split('_')[-1]  # z.B. "0005"
    speaker_id = suffix[-2:]             # z.B. "05"
    return speaker_id


def split_test_validation(
    base_dir: str,
    test_ratio: float = 0.8,
    seed: int = 42,
    audio_extensions=(".wav", ".flac", ".mp3", ".ogg"),
    count : int = None
) -> Tuple[List[Sample], List[Sample]]:
    """
    Liest den Ordner '50_speaker_audio_data' ein und splittet pro Speaker
    in ein Test- und ein Validation-Set.
    
    :param base_dir: Pfad zu '50_speaker_audio_data'
    :param test_ratio: Anteil der Dateien, die im Test-Set landen (Rest = Validation)
    :param seed: Zufalls-Seed für reproduzierbare Splits
    :param audio_extensions: akzeptierte Audio-Endungen
    :return: (test_set, validation_set), jeweils Liste von (audio_path, speaker_id)
    """
    rng = random.Random(seed)

    # Dateien pro Speaker sammeln
    speaker_to_files: Dict[str, List[str]] = {}

    for folder_name in sorted(os.listdir(base_dir)):
        speaker_folder = os.path.join(base_dir, folder_name)
        if not os.path.isdir(speaker_folder):
            continue
        speaker_id = parse_speaker_id_from_folder(folder_name)
        for filename in os.listdir(speaker_folder):
        
            if filename.lower().endswith(audio_extensions):
                full_path = os.path.join(speaker_folder, filename)
                speaker_to_files.setdefault(speaker_id, []).append(full_path)

    test_set: List[Sample] = []
    val_set: List[Sample] = []

    # Pro Speaker splitten, damit jeder Speaker in beiden Splits vorkommt (wenn genug Dateien)
    for speaker_id, files in speaker_to_files.items():
        rng.shuffle(files)
        n_total = len(files)
        if count is None:
            n_test = max(1, int(round(n_total * test_ratio)))  # mindestens 1 Datei im Test-Set
        else:
            n_test = count
        
        
        test_files = files[:n_test]
        val_files = files[n_test:]

        test_set.extend((f, speaker_id) for f in test_files)
        val_set.extend((f, speaker_id) for f in val_files)

    return test_set, val_set


def apply_on_test_set(
    test_set: List[Sample],
    fn: Callable[[str, str], Any]
) -> List[Any]:
    """
    Iteriert über das Test-Set und ruft fn(audio_path, speaker_id) auf.
    :param test_set: Liste von (audio_path, speaker_id)
    :param fn: Funktion, die z.B. Features extrahiert, Embeddings speichert, etc.
    :return: Liste der Rückgabewerte von fn (falls benötigt)
    """
    results = []
    for audio_path, speaker_id in test_set:
        result = fn(audio_path, speaker_id)
        results.append(result)
    return results


def evaluate_on_validation(
    val_set: List[Sample],
    predict_fn: Callable[[str], str],
  
) -> float:
    """
    Iteriert über das Validation-Set und ruft predict_fn(audio_path) auf.
    Die Funktion muss die ID der identifizierten Person (z.B. "05") zurückgeben.
    Am Ende wird die Accuracy berechnet.
    
    :param val_set: Liste von (audio_path, true_speaker_id)
    :param predict_fn: Funktion, die aus einer Audiodatei eine Speaker-ID vorhersagt
    :return: Accuracy (zwischen 0.0 und 1.0)
    """
    correct = 0
    total = 0
    mistakes = []
    for audio_path, true_id in val_set:
        pred_id = predict_fn(audio_path)
        if pred_id == true_id:
            correct += 1
        else:
            mistakes.append([pred_id,true_id])
        total += 1

    if total == 0:
        return 0.0
    
    allIDs = list(range(51))
    falsepositive = len(mistakes)
    countIDs(mistakes,allIDs) 
    falseidentifications = len([a for a in mistakes if a[0] is not None])
    print(f"FalseIdentifications: {((falseidentifications/total)*100):.2f}% ({falseidentifications}/{total})")
    accuracy = (correct / total) * 100 if total > 0 else 0
    i = (falsepositive / total) * 100 if total > 0 else 0
    print(f"\n✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"\n Mistakes: {i:.2f}% ({falsepositive}/{total})")




def func():
    for root, dirs, files in os.walk(input_root):
        for file in tqdm(files, desc=f"Processing {root}", leave=False):
            if file.endswith(".flac"):
                flac_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_root)
                output_dir = os.path.join(output_root, rel_path)
                os.makedirs(output_dir, exist_ok=True)

                wav_path = os.path.join(output_dir, file.replace(".flac", ".wav"))

                # Load and convert
                sound = AudioSegment.from_file(flac_path, format="flac")
                sound = sound.set_frame_rate(16000)
                sound.export(wav_path, format="wav")


def split_librispeech(
    input_root: str,
    output_root: str,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    count : int = None
):
    """
    Splits LibriSpeech-style directory structure into train and validation sets. https://openslr.org/31/

    Args:
        input_root (str): Path to LibriSpeech_wav folder.
        output_root (str): Path where split data will be stored.
        train_ratio (float): Fraction of files for training.
        shuffle (bool): Whether to shuffle before splitting.
    """
    for root, dirs, files in os.walk(input_root):
        wav_files = [os.path.join(root, f) for f in files if f.endswith(".wav")]
        if not wav_files:
            continue

        # Shuffle to randomize split
        if shuffle:
            random.shuffle(wav_files)

        # Split into train/val
        #
        if count is None:
            n_train = int(len(wav_files) * train_ratio)
        else:
            n_train = count
        train_files = wav_files[:n_train]
        val_files = wav_files[n_train:]

        # Copy to train folder
        for split_name, file_list in zip(["train", "val"], [train_files, val_files]):
            for f in file_list:
                rel_path = os.path.relpath(f, input_root)
                dest = os.path.join(output_root, split_name, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(f, dest)


        #tqdm.write(f"Processed {root}: {len(train_files)} train, {len(val_files)} val")

def default_speaker_id_extractor(root_path: str, split_root: str) -> str:
    """
    Given the directory containing wav files (root_path) and the split root (train/val root),
    return the speaker id. Works for structures like:
      split_root/train-clean-5/<speaker_id>/...
    It finds the first path component under split_root that consists only of digits,
    and returns that as the speaker id.
    """
    rel = os.path.relpath(root_path, split_root)
    parts = rel.split(os.sep)
    for p in parts:
        if p.isdigit():
            return p
    # fallback: return the first non-empty part
    for p in parts:
        if p:
            return p
    return ""

def evaluate_speaker_recognition(
    dataset_root: str,
    train_func,
    eval_func,
    train_subdir: str = "train",
    val_subdir: str = "val",
    train : bool = True,
    validate : bool = True
):
    """
    Traverses LibriSpeech_split train and val sets.
    Applies train_func to each training sample and eval_func to validation samples.
    Returns overall recognition accuracy.

    Args:
        dataset_root (str): Root directory of LibriSpeech_split.
        train_func (Callable): Function(train_wav_path, speaker_id) -> None
        eval_func (Callable): Function(val_wav_path) -> (bool, predicted_id)
        train_subdir (str): Name of training folder.
        val_subdir (str): Name of validation folder.

    Returns:
        float: Accuracy (percentage of correct recognitions).
    """
    #shutDownVoicerecognition()
    #initVoicerecognition()

    train_root = os.path.join(dataset_root, train_subdir)
    val_root = os.path.join(dataset_root, val_subdir)
    if train:
    # --- TRAIN ---
        #print("🔹 Training phase...")
        for root, _, files in os.walk(train_root):
            for f in tqdm(files, desc=f"Processing {root}", leave=False):
                if not f.endswith(".wav"):
                    continue
                wav_path = os.path.join(root, f)
                # Extract speaker ID (assumes dataset_root/train/<speaker_id>/...)
                speaker_id = default_speaker_id_extractor(root, train_root)
                if(int(speaker_id)<=50): break
                train_func(wav_path, speaker_id)

    if validate:
        # --- VALIDATION ---
        #print("🔹 Validation phase...")
        total = 0
        correct = 0
        mistakes = []
        listofIDs = []
        for root, _, files in os.walk(val_root):
            for f in tqdm(files, leave=False):
                if not f.endswith(".wav"):
                    continue
                wav_path = os.path.join(root, f)
                true_id = default_speaker_id_extractor(root, val_root)
                if true_id not in listofIDs: listofIDs.append(true_id)
                if int(true_id)<=50:break
                predicted_id = eval_func(wav_path)
                total += 1
                if predicted_id == true_id:
                    correct += 1
                    continue
                else:
                    mistakes.append([predicted_id,true_id])

        falsepositive = len(mistakes)
        countIDs(mistakes,listofIDs)
        falseidentifications = len([a for a in mistakes if a[0] is not None])
        print(f"FalseIdentifications: {((falseidentifications/total)*100):.2f}% ({falseidentifications}/{total})")      
        accuracy = (correct / total) * 100 if total > 0 else 0
        i = (falsepositive / total) * 100 if total > 0 else 0
        print(f"\n✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"\n Mistakes: {i:.2f}% ({falsepositive}/{total})")
        return accuracy

def countIDs(list,allIDs):
    count = {}
    for _,y in list:
        count[y] = count.get(y,0)+1
    print(f"IDs that were Missidentified:({len(count)}/{len(allIDs)})")
    return [[y,count.get(y,0)] for y in allIDs]


def runlibrispeech(dataset_root,train,test,vr):
    evaluate_speaker_recognition(dataset_root=dataset_root,train_func=vr.addFile,eval_func=vr.checkFile,train = train,validate=test)

def run50Speakers(vr,count):
    base_dir = "SpeakerRecognitionTests/50_speakers_audio_data"
    test_set, _ = split_test_validation(base_dir, test_ratio=0.8,count=count)
    apply_on_test_set(test_set, vr.addFile)

def validate50Speakers(vr,count):
    base_dir = "SpeakerRecognitionTests/50_speakers_audio_data"
    _, val_set = split_test_validation(base_dir, test_ratio=0.8,count=count)
    evaluate_on_validation(val_set, vr.checkFile)

#func()
#split_librispeech(input_root=output_root,output_root=split_root)
def resetVoicerecognitionAndDatasets(vr):
    vr.shutDownVoicerecognition()
    vr.initVoicerecognition()
    if Path.exists(split_root):
        shutil.rmtree(split_root)
    if Path.exists(split_root1):
        shutil.rmtree(split_root1)


def CompleteTrainingcycle(vr,count,resetIndex):
    if resetIndex:
        resetVoicerecognitionAndDatasets(vr)
        split_librispeech(input_root=output_root,output_root=split_root,count=count)
        split_librispeech(input_root=output_root1,output_root=split_root1,count = count)
        #Fill index
        runlibrispeech(split_root,True,False,vr)
        runlibrispeech(split_root1,True,False,vr)
        run50Speakers(vr,count=count)

    #Evaluate
    print("LibrispeechMini\n")
    runlibrispeech(split_root,False,True,vr)
    print("Librispeech\n")
    runlibrispeech(split_root1,False,True,vr)
    print("50Speakers\n")
    validate50Speakers(vr,count)

def runFullExperiments():
    #Maximum Case no add
    vr = SpeakerIdentificationSystem()
    print("\nTrainingcycle Minimum Case no add\n")
    vr.add = False
    CompleteTrainingcycle(vr,5,True)
    print("\nTrainingcycle Minimum Case no add Majority Vote\n")
    vr.majorityVote = True
    CompleteTrainingcycle(vr,5,False)
    vr.majorityVote = False
    print("\nTrainingcycle Maximum Case no add\n")
    CompleteTrainingcycle(vr,None,True)
    vr.majorityVote = True
    print("\nTrainingcycle Maximum Case no add MajorityVote\n")
    CompleteTrainingcycle(vr,None,False)
    vr.add = True
    vr.majorityVote = False
    print("\nTrainingcycle MinimumCase Add\n")
    CompleteTrainingcycle(vr,5,True)
    print("\nTrainingcycle Minimumcase Add MajorityVote\n")
    vr.majorityVote = True
    CompleteTrainingcycle(vr,5,True)
    vr.majorityVote = False
    print("\nTrainingcycle Maximumcase Add\n")
    CompleteTrainingcycle(vr,None,True)
    vr.majorityVote = True
    
    print("\nTrainingcycle Maximumcase Add Majorityvote\n")
    CompleteTrainingcycle(vr,None,True)
    

runFullExperiments()






