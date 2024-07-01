import pandas as pd
import pyreadstat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.stats import pearsonr

# Datei Pfad
file_path = '/Users/ahmadiqbalmss/Downloads/Datensatz_Stadtimage_Braunschweig_SoSe24_codiert.sav'

# SPSS .sav Datei laden
df, meta = pyreadstat.read_sav(file_path)

# Spaltennamen anzeigen, um die richtigen zu identifizieren
print(df.columns)

# Fälle entfernen, die den Aufmerksamkeitstest nicht bestanden haben
attention_test_var = 'EvBS_Kontrolle1'
df = df[df[attention_test_var] == 4]

# Konstrukte basierend auf den tatsächlichen Spaltennamen definieren
# Bitte ersetzen Sie die Platzhalter durch die tatsächlichen Spaltennamen
constructs_city_attractiveness = ['CICBS_CICB1', 'CICBS_CICB2', 'CICBS_CICB3', 'CICBS_CICB4']
constructs_recommend_city = ['NPSBS_NPSBS1', 'NPSBS_NPSBS2', 'NPSBS_NPSBS3']
constructs_cultural_offerings = ['EvBS_EvBS1', 'EvBS_EvBS2', 'EvBS_EvBS3']
constructs_likelihood_stay = ['ItSBS_ItSBS1', 'ItSBS_ItSBS2', 'ItSBS_ItSBS3']
constructs_city_image = ['PIBS_PIBS1', 'PIBS_PIBS2', 'PIBS_PIBS3']
economic_scientific_progress = ['BEBS_BEBS1', 'BEBS_BEBS2', 'BEBS_BEBS3']
constructs_infrastructure = ['InfBS_InfBS1', 'InfBS_InfBS2', 'InfBS_InfBS3', 'InfBS_InfBS4', 'InfBS_InfBS5']
constructs_likelihood_visit = ['ItVBS_ItVBS1', 'ItVBS_ItVBS2', 'ItVBS_ItVBS3']

# Faktorenanalyse für die Attraktivität der Stadt durchführen
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[constructs_city_attractiveness])
pca = PCA(n_components=1)
factor_scores = pca.fit_transform(scaled_data)
df['factor_city_attractiveness'] = factor_scores

# Reliabilitätsanalyse für die Attraktivität der Stadt
def cronbach_alpha(items):
    item_vars = items.var(axis=1, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    n_items = items.shape[1]
    alpha = n_items / (n_items - 1) * (1 - item_vars.sum() / total_var)
    return alpha

alpha_city_attractiveness = cronbach_alpha(df[constructs_city_attractiveness])
print(f'Cronbach\'s alpha für Stadtattraktivität: {alpha_city_attractiveness}')

# Hypothesentests mit Regressionsanalyse
# Hypothese 1: Attraktivität der Stadt -> Empfehlungswahrscheinlichkeit
X = df[['factor_city_attractiveness']]  # Unabhängige Variable
y = df[constructs_recommend_city].mean(axis=1)  # Abhängige Variable
X = sm.add_constant(X)  # Füge einen Konstantenterm zum Prädiktor hinzu
model = sm.OLS(y, X).fit()
print(model.summary())

# Faktorenanalyse für kulturelle Angebote
scaled_data_cultural = scaler.fit_transform(df[constructs_cultural_offerings])
pca_cultural = PCA(n_components=1)
factor_scores_cultural = pca_cultural.fit_transform(scaled_data_cultural)
df['factor_cultural_offerings'] = factor_scores_cultural

# Reliabilitätsanalyse für kulturelle Angebote
alpha_cultural_offerings = cronbach_alpha(df[constructs_cultural_offerings])
print(f'Cronbach\'s alpha für kulturelle Angebote: {alpha_cultural_offerings}')

# Hypothesentests mit Regressionsanalyse
# Hypothese 2: Kulturelle Angebote -> Wahrscheinlichkeit, in Braunschweig zu bleiben
X_cultural = df[['factor_cultural_offerings']]  # Unabhängige Variable
y_stay = df[constructs_likelihood_stay].mean(axis=1)  # Abhängige Variable
X_cultural = sm.add_constant(X_cultural)  # Füge einen Konstantenterm zum Prädiktor hinzu
model_cultural = sm.OLS(y_stay, X_cultural).fit()
print(model_cultural.summary())

# Faktorenanalyse für wirtschaftlichen und wissenschaftlichen Fortschritt
scaled_data_progress = scaler.fit_transform(df[economic_scientific_progress])
pca_progress = PCA(n_components=1)
factor_scores_progress = pca_progress.fit_transform(scaled_data_progress)
df['factor_progress'] = factor_scores_progress

# Reliabilitätsanalyse für wirtschaftlichen und wissenschaftlichen Fortschritt
alpha_progress = cronbach_alpha(df[economic_scientific_progress])
print(f'Cronbach\'s alpha für Fortschritt: {alpha_progress}')

# Hypothesentests mit Regressionsanalyse
# Hypothese 3: Fortschritt -> Stadtbildwahrnehmungen
X_progress = df[['factor_progress']]  # Unabhängige Variable
y_city_image = df[constructs_city_image].mean(axis=1)  # Abhängige Variable
X_progress = sm.add_constant(X_progress)  # Füge einen Konstantenterm zum Prädiktor hinzu
model_progress = sm.OLS(y_city_image, X_progress).fit()
print(model_progress.summary())

# Faktorenanalyse für Infrastruktur
scaled_data_infrastructure = scaler.fit_transform(df[constructs_infrastructure])
pca_infrastructure = PCA(n_components=1)
factor_scores_infrastructure = pca_infrastructure.fit_transform(scaled_data_infrastructure)
df['factor_infrastructure'] = factor_scores_infrastructure

# Reliabilitätsanalyse für Infrastruktur
alpha_infrastructure = cronbach_alpha(df[constructs_infrastructure])
print(f'Cronbach\'s alpha für Infrastruktur: {alpha_infrastructure}')

# Hypothesentests mit Regressionsanalyse
# Hypothese 4: Infrastruktur -> Besuchswahrscheinlichkeit
X_infrastructure = df[['factor_infrastructure']]  # Unabhängige Variable
y_visit = df[constructs_likelihood_visit].mean(axis=1)  # Abhängige Variable
X_infrastructure = sm.add_constant(X_infrastructure)  # Füge einen Konstantenterm zum Prädiktor hinzu
model_infrastructure = sm.OLS(y_visit, X_infrastructure).fit()
print(model_infrastructure.summary())

# Korrelationsanalyse
correlation_matrix = df[constructs_city_attractiveness + constructs_recommend_city + constructs_cultural_offerings + constructs_likelihood_stay + constructs_city_image + economic_scientific_progress + constructs_infrastructure + constructs_likelihood_visit].corr()
print(correlation_matrix)

# Pearson-Korrelation zwischen Konstrukten
for var1 in constructs_city_attractiveness:
    for var2 in constructs_recommend_city:
        if var1 != var2:
            corr, p_value = pearsonr(df[var1], df[var2])
            print(f'Korrelation zwischen {var1} und {var2}: {corr} (p-Wert: {p_value})')