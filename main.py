from email.policy import default
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
import os
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
import pickle
from nltk.corpus import stopwords
import ssl
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame, array
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

############################################

def generateNewKey():
    # Генерируете новый ключ (или берете ранее сгенерированный)
    key = RSA.generate(1024, os.urandom)
    return key

def getNewHashOf(pathFile: str):
    # Получаете хэш файла
    h = SHA256.new()

    with open(pathFile, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)

    print(h)

    return h

def sign(hash, key):
    # Подписываете хэш
    signature = pkcs1_15.new(key).sign(hash)
    return signature

def getPublicKey(key):
    # Получаете открытый ключ из закрытого
    pubkey = key.publickey()
    return pubkey

def savePubkeyAndSignature(pubkey, signature):
    f = open('pubkey.pem','wb')
    f.write(pubkey.exportKey('PEM'))
    f.close()

    with open('signature', 'wb') as picklefile:
        pickle.dump((signature), picklefile)

def fetchPubkeyAndSignatureFrom(path: str):
    f = open(f'{ path }/pubkey.pem','rb')
    pubkey = RSA.import_key(f.read())

    with open(f'{ path }/signature', 'rb') as training_model:
        signature = pickle.load(training_model)
    
    return (pubkey, signature)

def getFileFrom(pathFile: str):
    try:
        file = open(pathFile, 'r')
        # print(file.read())
    except Exception as e:
        print('\nSomething went wrong... Error: %s' % (e))

    return file

def checkSignature(hash, pubkey, signature):
    # Пересылаете пользователю файл, публичный ключ и подпись
    # На стороне пользователя заново вычисляете хэш файла (опущено) и сверяете подпись
    try: 
        pkcs1_15.new(pubkey).verify(hash, signature)
        return True
    except ValueError as error:
        print(f"Error: { error }")
        return False

    # Отличающийся хэш не должен проходить проверку
    # pkcs1_15.new(pubkey).verify(SHA256.new(b'test'), signature) # raise ValueError("Invalid signature")

############################################


def saveClassifierAndVectorizer(cl: RandomForestClassifier, vec: CountVectorizer):
    with open('text_classifier', 'wb') as picklefile:
        pickle.dump((cl, vec), picklefile)


def fetchClassifierAndVectorizer():
    with open('text_classifier', 'rb') as training_model:
        model, vectorizer = pickle.load(training_model)
    return (model, vectorizer)


def getTypeOfDocument(model: RandomForestClassifier, vec: CountVectorizer, data: list.__str__):
    X_train_counts = vec.transform(data).toarray()
    predicted = model.predict(X_train_counts)

    # print(predicted)

    return predicted


def loadFilesForMachineLearning(path: str):
    movie_data = load_files(path)
    data, target = movie_data.data, movie_data.target
    return (data, target)


def lemmatization(string: str):
    documents = []

    stemmer = WordNetLemmatizer()

    for sen in range(0, len(string)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(string[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)

    return documents


def getCountVectorizeAndVectorizer(documents: list.__str__):
    vectorizer = CountVectorizer(
        max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    data = vectorizer.fit_transform(documents).toarray()
    return data, vectorizer


def getTfidfTransformAndTfidfTransformer(documents: list.__str__):
    tfidfconverter = TfidfTransformer()
    data = tfidfconverter.fit_transform(documents).toarray()
    return data, tfidfconverter


def splitTrainAndTest(data: list.__str__, target: list.__str__):
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=0)
    return (X_train, X_test, y_train, y_test)


def classify(X_train: list.__str__, y_train: list.__str__):
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(X_train, y_train)
    return classifier

############################################

def whoIAm():
    print("Выберите кто Вы: \n1) Отправитель \n2) Получатель \n3) Переобучить модель")
    number = input("Введите номер: ")

    if number == "1":
        sender()
    elif number == "2":
        recipient()
    elif number == "3":
        retrainTheModel()
    else:
        print("Было введено неверное значение, попробуйте снова... \n")
        whoIAm()

def retrainTheModel():
    path = input("\nЗадайте путь к директории для обучения: ")
    
    data, target = loadFilesForMachineLearning(path)

    documents = lemmatization(data)
    documents, vectorizer = getCountVectorizeAndVectorizer(documents)
    documents, tfidfTransformer = getTfidfTransformAndTfidfTransformer(documents)

    X_train, X_test, y_train, y_test = splitTrainAndTest(documents, target)

    classifier = classify(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print(y_pred)

    saveClassifierAndVectorizer(classifier, vectorizer)

    print("Модель была успешно обучена и сохранена.")

def sender():
    pathFile = input("\nЗадайте путь к файлу: ")
    file = getFileFrom(pathFile)

    hash = getNewHashOf(pathFile)
    key = generateNewKey()
    
    signature = sign(hash, key)
    pubkey = getPublicKey(key)
    savePubkeyAndSignature(pubkey, signature)

    print("Публичный ключ и подпись были сохранены. Отправьте их получателю вместе с выбранным файлом.")

def recipient():
    pathFile = input("\nЗадайте путь к файлу: ")
    path = input("Задайте путь к директории публичного ключа и подписи: ")
    file = getFileFrom(pathFile)

    hash = getNewHashOf(pathFile)   
    pubkey, signature = fetchPubkeyAndSignatureFrom(path)

    result = checkSignature(hash, pubkey, signature)

    if result == True:
        verifiedMenu(pathFile)

def verifiedMenu(pathFile: str):
    print("\n1) Получить тип документа \n2) Создать отчёт")
    number = input("Введите номер: ")

    if number == "1":
        documentType(pathFile)
    elif number == "2":
        whoWillGetDocument(pathFile)
    else:
        print("Было введено неверное значение, попробуйте снова... \n")
        verifiedMenu()

def whoWillGetDocument(pathFile: str):
    print("Для кого сформировать отчёт? \n1) Руководство \n2) Бухгалтерия \n3) Сотрудник")
    number = input("Введите номер: ")

    documentType, txt = getDocumentTypeAndDocument(pathFile)

    if number == "1":
        print("Куда сохранить отчёт?")
        path = input()

        f = open(f"{ path }/Отчёт для руководства.txt ", mode='w', encoding='utf8')
        f.write(txt)

        print("Отчёт был успешно сформирован и сохранён")
    elif number == "2":
        if documentType[0] == 0:
            print("Куда сохранить отчёт?")
            path = input()
            
            f = open(f"{ path }/Отчёт для бухгалтерии.txt ", mode='w', encoding='utf8')
            f.write(txt)

            print("Отчёт был успешно сформирован и сохранён")
        elif documentType[0] == 1:
            print("Для бухгалтерии недоступен данный тип документа")
        else:
            print("Для бухгалтерии недоступен данный тип документа")
    elif number == "3":
        if documentType[0] == 0:
            print("Для сотрудника недоступен данный тип документа")
        elif documentType[0] == 1:
            print("Для сотрудника недоступен данный тип документа")
        else:
            print("Для сотрудника недоступен данный тип документа")
    else:
        print("Было введено неверное значение, попробуйте снова... \n")
        whoWillGetDocument(pathFile)

def documentType(path):
    documentType, _ = getDocumentTypeAndDocument(path)

    if documentType[0] == 0:
        print("Тип документа: Приказы, договоры, инструкции, служебные записки, протоколы")
    elif documentType[0] == 1:
        print("Тип документа: Организационно-распорядительные, финансово-отчетные, кадровые")
    else:
        print("Не удалось распознать тип документа")

    verifiedMenu(path)

def getDocumentTypeAndDocument(pathFile):
    examples = []

    f = open(pathFile, mode='r', encoding='utf8')
    txt = f.read()

    examples.append(txt)

    model, vectorizer = fetchClassifierAndVectorizer()
    
    documentType = getTypeOfDocument(model, vectorizer, examples)

    return documentType, txt

def main():
    whoIAm()

if __name__ == "__main__":
    main()