"""Check the retained final evidence, workbooks and document figures."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from .paths import ROOT, manifest, original_key

PUBLICATION = ROOT / "paper"
X = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
W = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
     "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_record(record):
    path = ROOT / record["path"]
    if not path.is_file() or ("bytes" in record and path.stat().st_size != record["bytes"]):
        raise ValueError("Missing or changed final file: " + str(path))
    if digest(path) != record["sha256"]:
        raise ValueError("Changed final file content: " + str(path))


def document_images(path):
    with ZipFile(path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        relations = {r.attrib["Id"]: r.attrib["Target"] for r in
                     ET.fromstring(archive.read("word/_rels/document.xml.rels"))}
        images = []
        for blip in document.findall(".//a:blip", W):
            target = relations[blip.attrib["{" + REL + "}embed"]]
            images.append(hashlib.sha256(archive.read("word/" + target)).hexdigest())
        # Text nodes exclude formatting, hyperlink IDs and drawing metadata.
        paragraphs = ["".join(t.text or "" for t in p.findall(".//w:t", W))
                      for p in document.findall(".//w:p", W)]
        return images, paragraphs


def verify_documents():
    """Match current embedded Word figures to their original PNG files."""
    profile=json.loads((PUBLICATION / "figure_manifest.json").read_text())
    results={}
    for stem,names in [("Manuscript",profile["main_figures"]),("Supplementary_Information",profile["supplementary_figures"])]:
        embedded,paragraphs=document_images(PUBLICATION / (stem+".docx"))
        expected=[digest(PUBLICATION / "figures" / (name+".png")) for name in names]
        if embedded!=expected:
            raise ValueError("Embedded figures differ from current source PNGs: "+stem)
        results[stem+".docx"]=dict(embedded_figures=len(embedded),extracted_figures=len(expected))
    return results


def verify_inventory():
    inventory = json.loads((ROOT / "repository_manifest.json").read_text())
    if inventory.get("schema_version") != 2:
        raise ValueError("Final repository inventory has not been generated")
    forcing = set()
    from models.climate.run import read_manifest
    forcing = {row["path"] for row in read_manifest()["forcing"]}
    records = inventory["files"]
    paths = [row["path"] for row in records]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate final inventory paths")
    for row in records:
        path = ROOT / row["path"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError("Final inventory file missing or changed: " + row["path"])
        if row["path"] not in forcing:
            check_record(row)
    return dict(files=len(records), content_hashes_verified=len(records) - len(forcing),
                final_forcing_sizes_verified=len(forcing),
                forcing_hash_scope="Frozen source hashes; use final_climate --verify-forcing-hashes to reread full forcing")


def verify(*, metrics=False, inventory=False):
    files = manifest()["files"]
    unique = {record["path"]: record for record in files.values()}
    for record in unique.values():
        check_record(record)
    from models.learned.integrity import verify as verify_models
    from models.climate.run import check_inputs
    model_contract = verify_models()
    climate = check_inputs()
    methods = json.loads((ROOT / "models/methods/source_manifest.json").read_text())
    for record in methods["files"]:
        check_record(record)
    result = dict(status="passed", retained_source_aliases=len(files),
        unique_evidence_files=len(unique), evidence_content_hashes_verified=True,
        documents=verify_documents(), portable_model_files=len(model_contract["files"]),
        scientific_recipe_files=len(methods["files"]), climate=climate)
    if metrics:
        import numpy as np
        import pandas as pd
        predictions=pd.read_csv(PUBLICATION/"data/fig4/predictions.csv")
        reference=pd.read_csv(PUBLICATION/"data/fig4/scores.csv").set_index("evaluation")
        scores={}
        for design,frame in predictions.groupby("evaluation"):
            w=frame.scientific_weight.to_numpy();y=frame.observed_percent.to_numpy();error=frame.prediction_percent.to_numpy()-y
            mse=np.average(error**2,weights=w);r2=1-mse/np.average((y-np.average(y,weights=w))**2,weights=w)
            rmse=float(np.sqrt(mse));np.testing.assert_allclose([rmse,r2],reference.loc[design,["rmse","r2"]].to_numpy(float),atol=1e-12,rtol=0)
            scores[design]=dict(n=len(frame),rmse=rmse,r2=float(r2))
        result["metrics"]=scores
    if inventory:
        result["repository_inventory"] = verify_inventory()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = json.dumps(verify(metrics=args.metrics, inventory=args.inventory), indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")
