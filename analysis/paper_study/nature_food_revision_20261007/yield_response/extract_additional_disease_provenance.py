"""Source-only audit of crop severity headers; no fitting or experimental protocols."""
from pathlib import Path
import re

from bs4 import BeautifulSoup
import pandas as pd

from verify_arithmetic import expanded_source_table

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/"yield_data_audit"


def assessment_date(text):
    iso = re.search(r"\d{4}-\d{2}-\d{2}",text)
    if iso:
        return iso.group()
    english = re.search(r"\d{1,2}/\d{1,2}/\d{4}",text)
    return pd.to_datetime(english.group(),format="%m/%d/%Y").date().isoformat() if english else None


def main():
    observations=[]
    sources=[("nfts_cohort_2022_67111.html","Wheat leaf spot % covered"),
             ("nfts_native_67111_sv.html","Svartpricksjuka % täckning"),
             ("nfts_native_67111_sv.html","Brunfläcksjuka (septoria) % täckning")]
    for filename,label in sources:
        soup=BeautifulSoup((SOURCE/filename).read_text(),"html.parser")
        for table_index,table in enumerate(soup.find_all("table")):
            if not (table.get("id") or "").startswith("tabResultatLedNieveau"):
                continue
            cells=expanded_source_table(table)
            columns=[c for (r,c),value in cells.items() if r==2 and value==label]
            for column in columns:
                date=assessment_date(cells.get((1,column),""))
                if date is None:
                    continue
                for row in sorted({r for r,c in cells if r>=3}):
                    factor,entry=cells.get((row,0),""),cells.get((row,1),"")
                    if factor not in {"A","B"} or not entry.isdigit():
                        continue
                    try:
                        value=float(cells.get((row,column),"").replace(",","."))
                    except ValueError:
                        continue
                    observations.append({"source_html":filename,"source_table":table_index,"source_table_id":table.get("id"),
                        "source_row":row,"source_column":column,"treatment_code":factor+entry,"date":date,
                        "measurement":label,"value":value,"unit":"Percent coverage","leaf_rank":"Unspecified"})
    observed=pd.DataFrame(observations).drop_duplicates(["source_html","treatment_code","date","measurement","value"])
    inputs=pd.read_csv(ROOT/"four_disease131_input.csv",dtype={"untreated_treatment_code":str,"treated_treatment_code":str})
    checks=[]
    for _,row in inputs[inputs.record=="nfts_67111"].iterrows():
        for arm,code in [("untreated",row.untreated_treatment_code),("treated",row.treated_treatment_code)]:
            native=observed[(observed.measurement=="Svartpricksjuka % täckning")&(observed.treatment_code==code)&(observed.date==row.date)]
            assert len(native)==1,(row.row_id,arm,len(native))
            value=float(native.value.iloc[0]);expected=float(row["stb" if arm=="untreated" else "stb_treated"])
            assert abs(value-expected)<1e-10,(row.row_id,value,expected)
            checks.append({"row_id":row.row_id,"arm":arm,"code":code,"date":row.date,"native_STB_value":value,"English_STB_input":expected})
    native_other=observed[observed.measurement=="Brunfläcksjuka (septoria) % täckning"]
    english_other=observed[observed.measurement=="Wheat leaf spot % covered"]
    joined=native_other.merge(english_other,on=["treatment_code","date"],suffixes=("_native","_English"),validate="one_to_one")
    assert len(joined)==164
    assert (joined.value_native==joined.value_English).all()
    observed.to_csv(ROOT/"additional_nordic_disease_provenance.csv",index=False)
    pd.DataFrame(checks).to_csv(ROOT/"native_STB_value_checks_67111.csv",index=False)
    joined.to_csv(ROOT/"native_nontarget_value_checks_67111.csv",index=False)
    print("Native STB final cells verified",len(checks))
    print("Native non-target disease cells matched to English",len(joined))
    print(english_other.groupby("date").value.agg(["count","min","max"]).to_string())


if __name__=="__main__":
    main()
