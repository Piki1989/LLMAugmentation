import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from transformers import pipeline
import openpyxl

# ==============================
# Parametry i funkcje pomocnicze
# ==============================
pkd_size = 272
rumowy_size = 4
grupowanie_size = 4
oddzial1_size = 24
branza1_size = 7
objlise_size = 7

wynik1 = 128
wynik2 = 64
wynik3 = 32


def simple_hash(val, size):
    return int(abs(hash(str(val))) % size)


# Inicjalizacja LLM – wykorzystujemy model "speakleash/Bielik-11B-v2.3-Instruct" do generacji tekstu
# Używamy pipelinu "text-generation" i ustawień do uzyskania parafrazy.
paraphraser = pipeline("text-generation", model="speakleash/Bielik-11B-v2.3-Instruct", max_length=50, do_sample=True,
                       temperature=0.7)
llm_cache = {}


def llm_paraphrase(text):
    """
    Wywołuje LLM, aby uzyskać parafrazę zadanego tekstu.
    Prompt: "Parafrazuj: <tekst>"
    Wykorzystuje cache do przyspieszenia.
    """
    if text in llm_cache:
        return llm_cache[text]
    prompt = f"Parafrazuj: {text}"
    result = paraphraser(prompt, max_length=50, num_return_sequences=1)
    paraphrased = result[0]['generated_text'].strip()
    llm_cache[text] = paraphrased
    return paraphrased


# ==============================
# Definicja architektury modelu
# ==============================
def prepare_model(X_pkd, X_rumowy, X_grupowanie, X_oddzial1, X_branza1, X_objlise):
    inputpkd = Input(shape=(len(X_pkd[0]),), name='in_pkd')
    pkd = Embedding(pkd_size, 10, input_length=1)(inputpkd)
    pkd_out = Flatten()(pkd)
    pkd_out = Dense(200, activation='relu')(pkd_out)

    inputrumowy = Input(shape=(len(X_rumowy[0]),), name='in_rodzaj_umowy')
    rumowy = Embedding(rumowy_size, 3, input_length=1)(inputrumowy)
    rumowy_out = Flatten()(rumowy)
    rumowy_out = Dense(100, activation='relu')(rumowy_out)

    inputgrupowanie = Input(shape=(len(X_grupowanie[0]),), name='in_grupowanie')
    grupowanie = Embedding(grupowanie_size, 3, input_length=1)(inputgrupowanie)
    grupowanie_out = Flatten()(grupowanie)
    grupowanie_out = Dense(100, activation='relu')(grupowanie_out)

    inputoddzial1 = Input(shape=(len(X_oddzial1[0]),), name='in_oddzial1')
    oddzial1 = Embedding(oddzial1_size, 10, input_length=1)(inputoddzial1)
    oddzial1_out = Flatten()(oddzial1)
    oddzial1_out = Dense(100, activation='relu')(oddzial1_out)

    inputbranza1 = Input(shape=(len(X_branza1[0]),), name='in_branza1')
    branza1 = Embedding(branza1_size, 4, input_length=1)(inputbranza1)
    branza1_out = Flatten()(branza1)
    branza1_out = Dense(100, activation='relu')(branza1_out)

    inputobjlise = Input(shape=(len(X_objlise[0]),), name='in_lise_object')
    objlise = Embedding(objlise_size, 4, input_length=1)(inputobjlise)
    objlise_out = Flatten()(objlise)
    objlise_out = Dense(100, activation='relu')(objlise_out)

    inputnumbers = Input(shape=(11,), name='in_continous')
    innum = Dense(200, activation="relu")(inputnumbers)
    innum = BatchNormalization()(innum)
    num_out = Dense(100, activation="relu")(innum)

    inputyesno = Input(shape=(3,), name='in_bool')
    inyesno = Dense(15, activation="relu")(inputyesno)
    yesno_out = Dense(10, activation="relu")(inyesno)

    merge = concatenate(
        [pkd_out, num_out, rumowy_out, grupowanie_out, yesno_out, oddzial1_out, branza1_out, objlise_out], axis=-1)

    hidden = Dense(wynik1, activation='relu')(merge)
    hidden = BatchNormalization()(hidden)
    hidden = Dense(wynik2, activation='relu')(hidden)
    hidden = Dense(wynik3, activation='relu')(hidden)
    output = Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=[inputpkd, inputnumbers, inputrumowy, inputgrupowanie, inputyesno, inputoddzial1, inputbranza1,
                          inputobjlise], outputs=output)
    return model


# ==============================
# Preprocessing danych uczących
# ==============================
dset = pd.read_csv('./data/probkauczaca2.csv', sep=';', encoding='utf-8')
Y = dset['Default'].values

vartype = pd.read_excel('./data/rodzaje_zmiennych.xlsx', engine='openpyxl')
columnX = []
X_binary = []
X_discrete = []
X_text_cat = []
for i in range(vartype.shape[1]):
    if vartype.iloc[0][i] == "numeric_continuous":
        columnX.append(dset.columns[i])
    if vartype.iloc[0][i] == "numeric_binary":
        X_binary.append(dset.columns[i])
    if vartype.iloc[0][i] == "numeric_discrete":
        X_discrete.append(dset.columns[i])
    if vartype.iloc[0][i] == "text_categorical":
        X_text_cat.append(dset.columns[i])

print('Kolumny numeryczne', columnX)
print('Kolumny binary numeryczne', X_binary)
print('Kolumny dyskretne numeryczne', X_discrete)
print('Kolumny tekstowe', X_text_cat)
print('Wszystkie kolumny w datasecie', dset.columns)

X = dset.loc[:, columnX].astype(float)
X['Kapital'] = X['Kapital'].fillna(0)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
X_standardized_df = pd.DataFrame(X_standardized, columns=X.columns)

scaler_minmax = MinMaxScaler()
for column_name in X_discrete:
    dset[column_name] = scaler_minmax.fit_transform(dset[[column_name]])

X_bin = dset.loc[:, ['Czy_cesja', 'Czy_leasing_zwrotny', 'Wynik_sprawdzen_KRD']].astype(str)
X_yesno = pd.DataFrame(X_bin['Czy_cesja'].eq('1').mul(1))
X_yesno['Czy_leas_zwr'] = X_bin['Czy_leasing_zwrotny'].astype(float)
X_yesno['Wyn_spraw_KRD'] = X_bin['Wynik_sprawdzen_KRD'].astype(float)

dset['kodpkd'] = dset['PKD_2007'].str.get(0) + dset['PKD_2007'].str.get(1) + dset['PKD_2007'].str.get(2)
dset['rodzumowy'] = dset['Rodzaj_umowy'].astype(str)
dset['oddzial1'] = dset['Oddzial'].astype(str)
dset['oddzial1'] = dset['oddzial1'].str.replace(' ', '')
dset['oddzial1'] = dset['oddzial1'].str.replace('-', '')
dset['oddzial1'] = dset['oddzial1'].str.strip()
dset['grupowanie'] = dset['Grupowanie_form_prawnych'].astype(str)
dset['branza1'] = dset['Branza'].astype(str)
dset['OBT_Type2'] = dset['OBT_Type2'].str.replace(' ', '')
dset['OBT_Type2'] = dset['OBT_Type2'].str.replace('&', '')

# ==============================
# Preprocessing danych testowych
# ==============================
tdset = pd.read_csv('./data/probkatestowa2.csv', sep=';', encoding='utf-8')
del tdset['Data_bilansowa']
tY = tdset['Default'].values

tX = tdset.loc[:, columnX].astype(float)
tX['Kapital'] = tX['Kapital'].fillna(0)
tscaler = StandardScaler()
tX_standardized = tscaler.fit_transform(tX)
tscaler_minmax = MinMaxScaler()
for column_name in X_discrete:
    tdset[column_name] = tscaler_minmax.fit_transform(tdset[[column_name]])

tX_bin = tdset.loc[:, ['Czy_cesja', 'Czy_leasing_zwrotny', 'Wynik_sprawdzen_KRD']].astype(str)
tX_yesno = pd.DataFrame()
tX_yesno['Czy_cesja'] = tX_bin['Czy_cesja'].eq('1').mul(1)
tX_yesno['Czy_leas_zwr'] = tX_bin['Czy_leasing_zwrotny'].astype(float)
tX_yesno['Wyn_spraw_KRD'] = tX_bin['Wynik_sprawdzen_KRD'].astype(float)

tdset['kodpkd'] = tdset['PKD_2007'].str.get(0) + tdset['PKD_2007'].str.get(1) + tdset['PKD_2007'].str.get(2)
tdset['rodzumowy'] = tdset['Rodzaj_umowy'].astype(str)
tdset['oddzial1'] = tdset['Oddzial'].astype(str)
tdset['oddzial1'] = tdset['oddzial1'].str.replace(' ', '')
tdset['oddzial1'] = tdset['oddzial1'].str.replace('-', '')
tdset['oddzial1'] = tdset['oddzial1'].str.strip()
tdset['grupowanie'] = tdset['Grupowanie_form_prawnych'].astype(str)
tdset['branza1'] = tdset['Branza'].astype(str)
tdset['OBT_Type2'] = dset['OBT_Type2'].str.replace(' ', '')
tdset['OBT_Type2'] = dset['OBT_Type2'].str.replace('&', '')

print(X.columns)

# Przygotowanie wejść do warstw Embedding dla danych testowych
X_pkd_test = np.array([simple_hash(v, pkd_size) for v in tdset['kodpkd']]).reshape(-1, 1)
X_rumowy_test = np.array([simple_hash(v, rumowy_size) for v in tdset['rodzumowy']]).reshape(-1, 1)
X_grupowanie_test = np.array([simple_hash(v, grupowanie_size) for v in tdset['grupowanie']]).reshape(-1, 1)
X_oddzial1_test = np.array([simple_hash(v, oddzial1_size) for v in tdset['oddzial1']]).reshape(-1, 1)
X_branza1_test = np.array([simple_hash(v, branza1_size) for v in tdset['branza1']]).reshape(-1, 1)
X_objlise_test = np.array([simple_hash(v, objlise_size) for v in tdset['OBT_Type2']]).reshape(-1, 1)

test_inputs = {
    'pkd': X_pkd_test,
    'numbers': tX_standardized,
    'rumowy': X_rumowy_test,
    'grupowanie': X_grupowanie_test,
    'yesno': tX_yesno.values,
    'oddzial1': X_oddzial1_test,
    'branza1': X_branza1_test,
    'objlise': X_objlise_test
}

test_text_fields = {
    'pkd': tdset['kodpkd'].tolist(),
    'rumowy': tdset['rodzumowy'].tolist(),
    'grupowanie': tdset['grupowanie'].tolist(),
    'oddzial1': tdset['oddzial1'].tolist(),
    'branza1': tdset['branza1'].tolist(),
    'objlise': tdset['OBT_Type2'].tolist()
}

y_test = tdset['Default'].values

# ==============================
# Budowa i kompilacja modelu
# ==============================
model = prepare_model(test_inputs['pkd'], test_inputs['rumowy'], test_inputs['grupowanie'],
                      test_inputs['oddzial1'], test_inputs['branza1'], test_inputs['objlise'])
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# ==============================
# Przygotowanie wejść do warstw Embedding dla danych uczących
# ==============================
X_pkd_train = np.array([simple_hash(v, pkd_size) for v in dset['kodpkd']]).reshape(-1, 1)
X_rumowy_train = np.array([simple_hash(v, rumowy_size) for v in dset['rodzumowy']]).reshape(-1, 1)
X_grupowanie_train = np.array([simple_hash(v, grupowanie_size) for v in dset['grupowanie']]).reshape(-1, 1)
X_oddzial1_train = np.array([simple_hash(v, oddzial1_size) for v in dset['oddzial1']]).reshape(-1, 1)
X_branza1_train = np.array([simple_hash(v, branza1_size) for v in dset['branza1']]).reshape(-1, 1)
X_objlise_train = np.array([simple_hash(v, objlise_size) for v in dset['OBT_Type2']]).reshape(-1, 1)

train_inputs = {
    'pkd': X_pkd_train,
    'numbers': X_standardized,
    'rumowy': X_rumowy_train,
    'grupowanie': X_grupowanie_train,
    'yesno': X_yesno.values,
    'oddzial1': X_oddzial1_train,
    'branza1': X_branza1_train,
    'objlise': X_objlise_train
}

# ==============================
# Trening modelu (model.fit)
# ==============================
model.fit([train_inputs['pkd'],
           train_inputs['numbers'],
           train_inputs['rumowy'],
           train_inputs['grupowanie'],
           train_inputs['yesno'],
           train_inputs['oddzial1'],
           train_inputs['branza1'],
           train_inputs['objlise']],
          Y,
          epochs=30,
          batch_size=32,
          validation_split=0.2)

# ==============================
# Ewaluacja na oryginalnych danych testowych
# ==============================
eval_clean = model.evaluate([test_inputs['pkd'],
                             test_inputs['numbers'],
                             test_inputs['rumowy'],
                             test_inputs['grupowanie'],
                             test_inputs['yesno'],
                             test_inputs['oddzial1'],
                             test_inputs['branza1'],
                             test_inputs['objlise']], y_test, verbose=0)
print("Ewaluacja na oryginalnych danych testowych:")
print("Loss: {:.4f}, Accuracy: {:.4f}".format(eval_clean[0], eval_clean[1]))


# ==============================
# Augmentacja danych testowych z wykorzystaniem LLM
# ==============================
# Dodanie szumu do zmiennych ciągłych (można zmienić scale)
# Modyfikacja zmiennych binarnych, flip z określonym prawdopodobieństwem (można zmienić rate)
# Augumentacja zmiennych tekstowych (parafraza, można zmienić prompt)
def augment_test_data_with_llm(data_dict, original_text_fields, perturbation_rate=0.1, scale=0.05):
    augmented = {}
    # Augmentacja zmiennych ciągłych – dodanie niewielkiego szumu
    augmented['numbers'] = data_dict['numbers'] + np.random.normal(0, scale, data_dict['numbers'].shape)

    # Augmentacja zmiennych binarnych – flip wartości z określonym prawdopodobieństwem
    augmented_yesno = []
    for row in data_dict['yesno']:
        new_row = []
        for val in row:
            new_row.append(1 - val if np.random.rand() < perturbation_rate else val)
        augmented_yesno.append(new_row)
    augmented['yesno'] = np.array(augmented_yesno)

    # Augmentacja zmiennych tekstowych – wywołanie LLM
    key_to_size = {
        'pkd': pkd_size,
        'rumowy': rumowy_size,
        'grupowanie': grupowanie_size,
        'oddzial1': oddzial1_size,
        'branza1': branza1_size,
        'objlise': objlise_size
    }
    for key in key_to_size.keys():
        size = key_to_size[key]
        new_vals = []
        orig_texts = original_text_fields[key]
        for i, orig_text in enumerate(orig_texts):
            if np.random.rand() < perturbation_rate:
                augmented_text = llm_paraphrase(orig_text)
                new_hash = simple_hash(augmented_text, size)
                new_vals.append(new_hash)
            else:
                new_vals.append(data_dict[key][i, 0])
        augmented[key] = np.array(new_vals).reshape(-1, 1)

    return augmented

# Wartości początkowe
perturbation_rate = 0.1
scale = 0.05

# Maksymalne wartości perturbation_rate i scale
max_perturbation_rate = 0.5
max_scale = 0.3

while perturbation_rate <= max_perturbation_rate and scale <= max_scale:
    print(f"\nTestowanie z perturbation_rate={perturbation_rate:.2f}, scale={scale:.2f}")

    test_inputs_aug = augment_test_data_with_llm(test_inputs, test_text_fields, perturbation_rate, scale)

    # ==============================
    # Ewaluacja na zaugmentowanych danych testowych
    # ==============================
    eval_aug = model.evaluate([test_inputs_aug['pkd'],
                               test_inputs_aug['numbers'],
                               test_inputs_aug['rumowy'],
                               test_inputs_aug['grupowanie'],
                               test_inputs_aug['yesno'],
                               test_inputs_aug['oddzial1'],
                               test_inputs_aug['branza1'],
                               test_inputs_aug['objlise']], y_test, verbose=0)

    print("\nEwaluacja na zaugmentowanych (LLM) danych testowych:")
    print("Loss: {:.4f}, Accuracy: {:.4f}".format(eval_aug[0], eval_aug[1]))

    print("\nPorównanie wyników:")
    print("Oryginalne dane - Loss: {:.4f}, Accuracy: {:.4f}".format(eval_clean[0], eval_clean[1]))
    print(
        "Zaugmentowane dane z parametrami perturbation_rate {:.4f}, scale {:.4f} - Loss: {:.4f}, Accuracy: {:.4f}".format(
            perturbation_rate, scale, eval_aug[0], eval_aug[1]))

    # Zwiększenie wartości parametrów
    perturbation_rate += 0.05
    scale += 0.05