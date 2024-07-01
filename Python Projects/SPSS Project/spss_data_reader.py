import pandas as pd
import pyreadstat

# Datei Pfad
file_path = '/Users/ahmadiqbalmss/Downloads/Datensatz_Stadtimage_Braunschweig_SoSe24_codiert.sav'

# SPSS .sav Datei laden
df, meta = pyreadstat.read_sav(file_path)

# Optionen setzen, um alle Spaltennamen anzuzeigen
pd.set_option('display.max_columns', None)

# Spaltennamen anzeigen, um die richtigen zu identifizieren
print(df.columns)

# Option zurücksetzen, falls nötig
pd.reset_option('display.max_columns')