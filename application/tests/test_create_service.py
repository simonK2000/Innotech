import requests
import ast

def test_service():
    prediction = requests.post(
        'http://0.0.0.0:5001/employee/predict',
        headers={'content-type': 'application/json'},
        data='{"City": "Pune", "PaymentTier": 0, "Age": 0, "Gender": "Female", "EverBenched": "No", "ExperienceInCurrentDomain": 0}',
    ).text
    print(prediction)
    print(type(prediction))
    res = ast.literal_eval(prediction)
    assert res['prediction'] in [0, 1]



