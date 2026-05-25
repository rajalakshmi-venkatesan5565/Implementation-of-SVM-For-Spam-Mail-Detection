# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.open new colab notebook.
2.Enter python code.
3. run the python code.
4. program run successfully.
5. take the screenshot of the output.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Rajalakshmi V
RegisterNumber:  212225040327
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Create sample dataset
data = {
    'label': [
        'ham','spam','ham','spam','ham',
        'spam','ham','spam','ham','spam'
    ],

    'message': [
        'Hey, how are you?',
        'Congratulations! You won a lottery',
        'Are you coming to class today?',
        'Claim your free prize now',
        'Meeting starts at 10 AM',
        'Click here and get cash reward',
        'Can you send notes?',
        'You have won a free vacation',
        'Assignment submission tomorrow',
        'Limited offer claim money now'
    ]
}

df = pd.DataFrame(data)

# Convert labels
df['label'] = df['label'].map({
    'ham':0,
    'spam':1
})

X = df['message']
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
svm_model = SVC(
    kernel='linear',
    random_state=42
)

svm_model.fit(X_train_vec, y_train)

# Predict
y_pred = svm_model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test,y_pred)

print("Model Accuracy:", round(accuracy*100,2), "%")

print("\nClassification Report:")
print(classification_report(y_test,y_pred))

# Custom email test
email = [
    "Congratulations! You have won a free lottery ticket"
]

email_vec = vectorizer.transform(email)

prediction = svm_model.predict(email_vec)

if prediction[0]==1:
    print("\nPrediction: Spam Mail")
else:
    print("\nPrediction: Not Spam")
```

## Output:
<img width="1906" height="933" alt="Screenshot 2026-05-25 131743" src="https://github.com/user-attachments/assets/4a308b3d-7b4e-4034-816e-34f2bd38993f" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
