import modal

app = modal.App("alphabet-stt")

image = modal.Image.debian_slim().pip_install("fastapi[standard]")

@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def hello():
    return {"message": "Alphabet STT service is alive"}