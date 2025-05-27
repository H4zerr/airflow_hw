# <YOUR_IMPORTS>
import os
from datetime import datetime

import json
import pandas as pd
import dill
from pydantic import BaseModel

folder_models = 'airflow_hw/data/models'
model_f = [f for f in os.listdir(folder_models)
           if f.startswith('cars_pipe') and f.endswith('pkl')
           ]

model_f.sort(reverse=True)
model_name = model_f[0]

with open(f'airflow_hw/data/models/{model_name}', 'rb') as f:
    model = dill.load(f)

folder = 'airflow_hw/data/test'
json_files = [f for f in os.listdir(folder) if f.endswith('.json')]

class Form(BaseModel):
    description: str
    fuel: str
    id: int
    image_url: str
    lat: float
    long: float
    manufacturer: str
    model: str
    odometer: int
    posting_date: str
    price: float
    region: str
    region_url: str
    state: str
    title_status: str
    transmission: str
    url: str
    year: int

timestamp = model_name.replace('cars_pipe_', '').replace('.pkl', '')


def predict():
    results = []

    for filename in json_files:
        file_path = os.path.join(folder, filename)

        with open(file_path, 'r') as f:
            raw_data = json.load(f)
            form = Form(**raw_data)

        data = pd.DataFrame.from_dict([form.model_dump()])
        prediction = model.predict(data)[0]
        results.append({
            'car_id' : form.id,
            'pred' : prediction
        })
    df = pd.DataFrame(results)
    df.to_csv(f'airflow_hw/data/predictions/prediction_{timestamp}.csv', index=False)


if __name__ == '__main__':
    predict()
