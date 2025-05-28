from fastapi import (
    FastApi,
    Form,
    Request,
    Response,
    File,
    Depends,
    HTTPException,
    status,
)
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder

import uvicorn
import os
import aiofiles
import json
import csv

from src.helper import llm_pipeline


def get_csv(file_path: str):
    answer_gen_chain, ques_list = llm_pipeline(file_path)
    base_folder = "static/output/"
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder + "QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])  # writing the header row

        for question in ques_list:
            print("Question: ", question)
            answer = answer_gen_chain.run(question)
            print("Answer: ", answer)
            print("-----------------------------")

            # Save answer to csv file
            csv_writer.writerow([question, answer])
    return output_file


app = FastApi()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

app.get("/")


async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("analyze")
async def chat(request: Request, pdf_filename: str = Form(...)):
    output_file = get_csv(pdf_filename)
    response_data = jsonable_encoder(json.dumps({"output_file": output_file}))
    res = Response(response_data)
    return res


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
