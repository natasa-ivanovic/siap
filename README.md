# Sentiment analiza recenzija hotela na srpskom i engleskom jeziku i izdvajanje aspekata

## Tema projekta
- Sentiment analiza recenzija hotela na srpskom i engleskom jeziku primenom tehnika mašinskog učenja
- Izdvajanje aspekata (engl. NER - Named entity recognition) u hotelskim recenzijama na srpskom i englkeskom jeziku primenom neuronskih mreža

## Implementacija
- Detekcija jezika - Multinomial Naive Bayes model
- Sentiment analiza na srpskom i engleskom jeziku - SVM i Random Forest modeli
- NER na srpskom jeziku - BiLSTM-CRF model
- NER na engleskom jeziku - prilagođeni SpaCy model

## Skup podataka
- 1044 engleskih recenzija preuzetih iz skupa podataka sa [sajta](https://www.kaggle.com/meetnagadia/hotel-reviews) za sentiment analizu
- Podskup od 500 engleskih recenzija za NER
- 1389 srpskih recenzija skrejpovanih sa [sajta](https://www.booking.com/reviews) za sentiment analizu
- Podskup od 300 srpskih recenzija za NER
- Unija skupova srpskih i engleskih recenzija za detekciju jezika

## Pokretanje projekta
- Za izradu projekta korišćen je Python 3.10 sa odgovarajućim bibliotekama
- Za pokretanje treniranja i testiranja detekcije jezika potrebno je pokrenuti fajl _language_detection/LanguageDetection.py_
- Za pokretanje trenirnja i testiranja sentiment analize sa SVM modelom potrebno je pokrenuti fajl _models/SVMModel.py_
- Za pokretanje trenirnja i testiranja sentiment analize sa Random Forest modelom potrebno je pokrenuti fajl _models/RandomForestModel.py_
- Za pokretanje trenirnja i testiranja NER-a sa srpskim recenzijama potrebno je pokrenuti fajl _BiLSTMCRF.py_
- Za pokretanje trenirnja i testiranja NER-a sa engleskim recenzijama potrebno je pokrenuti fajl _CustomSpacy.py_

## Rezultati
- Rezultati testiranja sentiment analize nad test skupom mogu se naći u folderu [models](https://github.com/natasa-ivanovic/siap/tree/main/models)
- Rezultati testiranja NER modela nad trening i test skupom mogu se naći u folderu [data/ner](https://github.com/natasa-ivanovic/siap/tree/main/data/ner)

## Članovi tima
- [Vera Kovačević R214/2021](https://github.com/verak13)
- [Nataša Ivanović R212/2021](https://github.com/natasa-ivanovic)

