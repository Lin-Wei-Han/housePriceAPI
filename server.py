from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import gzip
import pandas as pd

app = FastAPI()


class HousePriceItem(BaseModel):
    buildingArea: float
    roomAmount: int
    livingroomAmount: int
    bathroomAmount: int


with gzip.open('randomForest_houseprice.pkl', 'rb') as f:
    housePriceModel = pickle.load(f)


@app.post('/getHousePrice')
async def housePrice_endpoint(item: HousePriceItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    result = housePriceModel.predict(df)
    return {"prediction": int(result)}
