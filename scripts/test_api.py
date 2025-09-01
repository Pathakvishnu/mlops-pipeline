import requests

url = "http://127.0.0.1:8000/predict"

payload = [
    {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
        "Name": "Braund, Mr. Owen Harris",
        "Ticket": "A/5 21171",
        "Cabin": ""
    },
    {
        "Pclass": 1,
        "Sex": "female",
        "Age": 60,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
        "Name": "Braund, Mr. Owen Harris",
        "Ticket": "A/5 21171",
        "Cabin": ""
    }
]

response = requests.post(url, json=payload)
print(response.json())