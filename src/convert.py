import pandas as pd
import boto3
import os

from digitalhub_runtime_python import handler

@handler()
def convert_spira(project, spira_di, name):
    # read and rewrite to normalize and export as data
    spira_csv = spira_di.as_df()
    spira_csv = spira_csv.rename(columns={'codice_spira': 'spira_code', 'chiave':'spira_unique_id'})
    
    spira_melted = spira_csv.melt(id_vars=['data', 'spira_unique_id', 'spira_code','longitudine', 'latitudine'],
                       value_vars=[col for col in spira_csv.columns if col.startswith(('00_', '01_', '02_', '03_', '04_', '05_', '06_',
                                                                               '07_', '08_', '09_', '10_', '11_', '12_', '13_',
                                                                               '14_', '15_', '16_', '17_', '18_', '19_', '20_',
                                                                               '21_', '22_', '23_'))],
                       var_name='hour',
                       value_name='count')
    
    spira_melted['DateTime'] = pd.to_datetime(spira_melted['data'].astype(str) + ' ' + spira_melted['hour'].str.replace('_', ':').str[:5] + ':00')
    spira_melted['DateTime'] = spira_melted['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')

          
    spira_melted.drop(columns=['data', 'hour'], inplace=True)
    
    project.log_dataitem(name=name, kind="table", data=spira_melted)

@handler()
def convert_spira_accur(project, spira_accur_di, name):
    accuracy_csv = spira_accur_di.as_df()
    accuracy_csv=accuracy_csv.rename(columns={'codice_spira_2': 'spira_code', 'data_2': 'data'})

    acc_melted = accuracy_csv.melt(id_vars=['data', 'spira_code'],
                   value_vars=[col for col in accuracy_csv.columns if col.startswith(('00_', '01_', '02_', '03_', '04_', '05_', '06_',
                                                                           '07_', '08_', '09_', '10_', '11_', '12_', '13_',
                                                                           '14_', '15_', '16_', '17_', '18_', '19_', '20_',
                                                                           '21_', '22_', '23_'))],
                   var_name='hour',
                   value_name='count')

    acc_melted['DateTime'] = pd.to_datetime(acc_melted['data'] + ' ' + acc_melted['hour'].str.replace('_', ':').str[:5] + ':00')
    acc_melted.drop(columns=['data', 'hour'], inplace=True)
    acc_melted['count'] = acc_melted['count'].str.replace('%', '').astype(int)
    
    acc_melted['DateTime'] = acc_melted['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    project.log_dataitem(name=name, kind="table", data=acc_melted)

def _convert(export):
    converted = []

    ir = export.iterrows()
    
    # Parse general header
    i,r = next(ir)
    assert r["Da"] == "Da"
    
    i,r = next(ir)
    while True:
        # Parse data row
        assert r["Da"] == r["Da"] 
        data = r["Da"]
        # Parse varco header
        i,r = next(ir)
        assert r["Da"] != r["Da"]
        assert r["A"] == "Varco" 
        # Parse varco section
        i,r = next(ir)
        while True:
            # Parse varco row
            if r["Da"] == r["Da"]: break
            assert r["A"] == r["A"]
            varco = r["A"]
            # Parse veicolo header
            i,r = next(ir)
            assert r["Da"] != r["Da"]
            assert r["A"] != r["A"]
            if (r["Dettaglio totali"] == "Motivo spostamento"):
                # Parse motivo spostamento
                i,r = next(ir)
                assert r["Dettaglio totali"] == "Transito Incompleto"
                i,r = next(ir)
            assert r["Dettaglio totali"] == "Classe veicoli"
            while True:
                # Parse veicolo row
                i,r = next(ir)
                if (r["Da"] == r["Da"]) or (r["A"] == r["A"]) or (r["Dettaglio totali"] != r["Dettaglio totali"]):
                    break
                assert r["Dettaglio totali"] != "Classe veicoli"
                veicolo = r["Dettaglio totali"]
                cont = r["Dettaglio totali.1"]
                converted.append({"Data": data, "Varco": varco, "Veicolo": veicolo, "Conteggio": cont})
            if r["Dettaglio totali"] != r["Dettaglio totali"]: break
        # Parse data trailer
        while True:
            try:
                i,r = next(ir)
                if (r["Da"] == r["Da"]) or (r["A"] == r["A"]): break
            except StopIteration:
                return pd.DataFrame(converted)

def read_gates(fname):
    gates = pd.read_excel(fname)
    gates = gates.iloc[:gates["Varco"].isnull().idxmax()]
    gates[["ID","Varco"]] = gates["Varco"].str.split('\\) ', n=1, expand=True)
    # Move ID column to the front
    gates = gates[gates.columns.tolist()[-1:] + gates.columns.tolist()[:-1]]
    gates.loc[gates["Varco"] == "Corelli Nord", "Varco"] = "Corelli_1 Nord"
    gates.loc[gates["Varco"] == "Corelli Sud", "Varco"] = "Corelli_2 Sud"
    gates.loc[gates["Varco"] == "Bonvicini", "Varco"] = "Porrettana/Bonvicini"
    gates["ID"] = gates["ID"].astype("string")
    gates["Varco"] = gates["Varco"].astype("string")
    gates["Indirizzo"] = gates["Indirizzo"].astype("string")
    gates["Settore"] = gates["Settore"].astype("string")
    gates["Link google maps"] = gates["Link google maps"].astype("string")
    gates = gates.rename(columns={'Varco':'gate'})
    return gates

def check(df):
    unique = df.groupby(['Data', 'gate', 'vehicle','count']).nunique().reset_index()
    multiple = unique.loc[unique.duplicated(keep=False, subset=['Data', 'gate', 'vehicle'])]
    if multiple.size:
        raise Exception(f"Multiple rows with different values ({multiple.shape[0]} rows)")
    if df.size != unique.size:
        raise Exception(f"Multiple rows with same values ({df.shape[0] - unique.shape[0]} rows)")


@handler(outputs=["gates", "gate_data"])
def create_gate_data(project, bucket, gates_file, data_files):

    s3 = boto3.client('s3',
                    endpoint_url=os.environ.get('S3_ENDPOINT_URL'),
                    aws_access_key_id=os.environ.get('PROTECTED_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.environ.get('PROTECTED_SECRET_ACCESS_KEY'))

    s3.download_file(bucket, gates_file, "./gates.xlsx")
    gates = read_gates("./gates.xlsx")

    merged = None
    for f in data_files:
        s3.download_file(bucket, f, "./data.xlsx")
        export = pd.read_excel("./data.xlsx")
        converted = _convert(export) 
        converted.loc[converted["Veicolo"] == "Ciclomotore ", "Veicolo"] = "Ciclomotore"
        converted["Data"] = pd.to_datetime(converted["Data"], format="%d/%m/%Y %H:%M").dt.strftime('%Y-%m-%d %H:%M:%S')
        converted["Varco"] = converted["Varco"].astype("string")
        converted["Veicolo"] = converted["Veicolo"].astype("string")
        converted["Conteggio"] = converted["Conteggio"].astype("int")
        converted = converted.rename(columns={'Varco':'gate', 'Veicolo':'vehicle', 'Conteggio': 'count'})
        if merged is None:
            merged = converted
        else:
            merged = pd.concat([merged, converted])

    check(merged)
    return gates, merged