from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from newsplease import NewsPlease
from pydantic import BaseModel

from app.model import load_model, get_embedding


class ClassificationReport(BaseModel):
    title: str
    confidence: float
    class_: str


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

model = load_model()


def get_article_info(url):
    article = NewsPlease.from_url(url)
    return article.title, article.maintext


@app.get("/predict", response_model=ClassificationReport)
async def predict_news(url: str):
    try:
        title, body = get_article_info(url)
    except Exception as e:
        return ClassificationReport(
            title="Failed to retrieve data",
            confidence=0.0,
            class_="Fail"
        )

    embedding = get_embedding(title + " " + body)
    result = model.predict(embedding)[0]
    result_class = "Fake"
    confidence = result[1]

    if result[0] > result[1]:
        result_class = "Not Fake"
        confidence = result[0]

    return ClassificationReport(
        title=title,
        confidence=float("{:.2f}".format(confidence * 100)),
        class_=result_class
    )
