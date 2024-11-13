import pandas as pd

PATHS = [
    'datasets/CICIDS-2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'datasets/CICIDS-2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'datasets/CICIDS-2017/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'datasets/CICIDS-2017/Monday-WorkingHours.pcap_ISCX.csv',
    'datasets/CICIDS-2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'datasets/CICIDS-2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'datasets/CICIDS-2017/Tuesday-WorkingHours.pcap_ISCX.csv',
    'datasets/CICIDS-2017/Wednesday-workingHours.pcap_ISCX.csv',
]

OUTPUT = 'datasets/CICIDS-2017/dataset.csv'

# Lista para armazenar os DataFrames
dataframes = []

# Carrega cada dataset e adiciona à lista
for path in PATHS:
    df = pd.read_csv(path)
    dataframes.append(df)

# Concatena todos os DataFrames em um único
concatenated_df = pd.concat(dataframes, ignore_index=True)
concatenated_df.columns = concatenated_df.columns.str.strip()

# Removendo espaços extras
concatenated_df.columns = concatenated_df.columns.str.strip()

# Trocando os valores da coluna label
filter = concatenated_df['Label'] == "BENIGN"
concatenated_df.loc[filter, 'Label'] = "0" 
concatenated_df.loc[~filter, 'Label'] = "1"

# Salva o DataFrame concatenado em um arquivo CSV
concatenated_df.to_csv(OUTPUT, index=False)