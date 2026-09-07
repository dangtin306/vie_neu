"""Create anonymized TEST13 blind triplets. Does not train or inspect scores."""
from __future__ import annotations
import csv, json, random, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"output"/"test13_scale_feasibility"
BLIND=OUT/"blind_listening"
MODELS=["base","tf","warm98_2"]
SEED=20250826

def main():
    summary=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    test=list(csv.DictReader((ROOT/"test_pairs.csv").open(encoding="utf-8-sig",newline="")))
    for d in [BLIND/"reference",BLIND/"A",BLIND/"B",BLIND/"C"]:
        d.mkdir(parents=True,exist_ok=True)
    rng=random.Random(SEED)
    private={}; rows=[]
    for i,row in enumerate(test):
        pair=f"pair_{i:02d}_{Path(row['target_path']).stem}"
        # The source WAVs use case_i_j_*; selecting the newest is deterministic for this run.
        for j in range(3):
            case=f"case_{i:03d}_{j+1:02d}"
            ref=Path(row["reference_path"])
            shutil.copy2(ref,BLIND/"reference"/(case+".wav"))
            order=MODELS[:]; rng.shuffle(order)
            entry={}
            for letter,model in zip("ABC",order):
                hits=sorted((OUT/"blind_audio"/model).glob(f"case_{i:02d}_{j:02d}_*.wav"))
                if not hits: raise FileNotFoundError(f"Missing {model} case {i}/{j}")
                shutil.copy2(hits[-1],BLIND/letter/(case+".wav"))
                entry[letter]=model
            private[case]={"speakerID":row["speakerID"],"pair_id":pair,"sentence_index":j+1,"reference":str(ref),"models":entry}
            r={"case_id":case,"speakerID":row["speakerID"],"pair_id":pair,"reference_file":str(BLIND/"reference"/(case+".wav"))}
            for c in ["accent","naturalness","rhythm","pronunciation","speaker_similarity"]:
                for x in "ABC": r[f"{x}_score_{c}"]=""
            r.update(accent_winner="",naturalness_winner="",notes="")
            rows.append(r)
    (BLIND/"mapping_private.json").write_text(json.dumps(private,ensure_ascii=False,indent=2),encoding="utf-8")
    fields=list(rows[0])
    with (BLIND/"score_sheet.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"Created {len(rows)} blind triplets in {BLIND}")
    print("Do not open mapping_private.json until scoring is complete.")
if __name__=="__main__": main()
