import requests
import json


def Complexity_Calc(file_path):

    #Change to fit the need
    threshold = 0.0000003
    url = 'https://books.google.com/ngrams/json'
    
    #Testing
    file_path = "Plan for today.txt"
    
    params = {
    'content': 'Churchill',
    'year_start': '1800',
    'year_end': '2000',
    'corpus': '26',
    'smoothing': '3'
    }
    
    # Open the file and read its contents
    with open(file_path, 'r') as file:
        contents = file.read()

    splits = contents.strip().split()

    total = 0
    for split in splits:
        params = {
        'content': split,
        'year_start': '2010',
        'year_end': '2019',
        'corpus': '26',
        'smoothing': '3'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        data = data[0]['timeseries']
        total += sum(data)
        break
    
    avg = total/(len(splits)*9)

    #complexity 0 - low, 1- med, 2 - high
    if avg < (threshold-(threshold*0.15)):
        complexity_rating = 2
    elif avg > (threshold+(threshold*0.15)):
        complexity_rating = 0
    else:
        complexity_rating = 1

    #Higher Value less complex
    return complexity_rating

complexity_rating = Complexity_Calc("Plan for today.txt")
print(complexity_rating)