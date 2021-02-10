import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from newsplease import NewsPlease
from pydantic import BaseModel

from app.model import load_model, process


class ClassificationReport(BaseModel):
    title: str
    confidence: float
    class_: str


app = FastAPI()
nltk.download("punkt")

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

model, char2idxs = load_model(weights_path="model_weights/model.h5", char2idx_path="model_weights/char2idxs.pkl")


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

    proc_title = process(title, char2idxs['title_char2idx'])
    proc_body = process(body, char2idxs['text_char2idx'])

    proc_title = proc_title.reshape(1, *proc_title.shape)
    proc_body = proc_body.reshape(1, *proc_body.shape)

    result = model.predict([proc_title, proc_body])[0]
    result_class = "Fake"
    confidence = result[0]
    if confidence < 0.5:
        result_class = "Not Fake"
        confidence = 1 - confidence

    return ClassificationReport(
        title=title,
        confidence=float("{:.2f}".format(confidence * 100)),
        class_=result_class
    )
