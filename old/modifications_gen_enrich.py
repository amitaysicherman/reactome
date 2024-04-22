# RUN in GCP workbench
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
import os


def generate(text1):
    vertexai.init(project="netsys-ai-xgcp", location="us-central1")
    model = GenerativeModel("gemini-1.0-pro-001")
    responses = model.generate_content(
        [text1],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )
    text = ""
    for response in responses:
        text += response.text
    return text


generation_config = {
    "max_output_tokens": 2048,
    "temperature": 0.9,
    "top_p": 1,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

input_file = "./data/items/modifications.txt"
output_file = "./data/items/modifications_enrich.txt"
if os.path.exists(output_file):
    os.remove(output_file)
prompt = "This is a term that describes biological modification, explain its meaning in a few sentences :"
with open(input_file) as f:
    modifications = f.read().splitlines()
for modification in modifications:
    text = modification.split(":")[0]
    print(text)
    for i in range(3):
        try:
            enrich_text = generate(f"{prompt}'{text}'")
            break
        except:
            print("retray", i)
    else:
        enrich_text = text

    with open(output_file, "a") as f:
        f.write(f"{enrich_text}\nAMITAY_END\n")
