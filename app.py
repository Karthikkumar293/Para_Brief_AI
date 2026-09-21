import os
import torch

from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# PARA BRIEF AI
# Backend API
# ============================================================

app = Flask(__name__, static_folder=".")


# ============================================================
# MODEL PATH
# ============================================================

# After extracting your Best_3 model, put the model folder here.
#
# Example:
#
# Para_Brief_AI/
# ├── app.py
# ├── index.html
# ├── requirements.txt
# └── best_3/
#     ├── config.json
#     ├── tokenizer_config.json
#     ├── tokenizer.json
#     ├── model.safetensors
#     └── ...
#

MODEL_PATH = os.environ.get("MODEL_PATH", "best_3")


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("PARABRIEF AI")
print("=" * 60)
print("Loading model...")
print("Model path:", MODEL_PATH)
print("Device:", device)


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


# ============================================================
# LOAD MODEL
# ============================================================

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()


print("Model loaded successfully.")
print("=" * 60)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return send_from_directory(".", "index.html")


# ============================================================
# SUMMARIZATION API
# ============================================================

@app.route("/summarize", methods=["POST"])
def summarize():

    try:

        # ----------------------------------------------------
        # Get JSON data
        # ----------------------------------------------------

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No request data received."
            }), 400


        text = data.get("text", "").strip()


        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if not text:
            return jsonify({
                "error": "Please enter some text to summarize."
            }), 400


        # ----------------------------------------------------
        # Tokenize input
        # ----------------------------------------------------

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )


        # Move tensors to device

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }


        # ----------------------------------------------------
        # Generate summary
        # ----------------------------------------------------

        with torch.no_grad():

            output_ids = model.generate(
                **inputs,

                max_new_tokens=100,

                min_new_tokens=20,

                num_beams=4,

                no_repeat_ngram_size=3,

                early_stopping=True
            )


        # ----------------------------------------------------
        # Decode generated tokens
        # ----------------------------------------------------

        summary = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        ).strip()


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return jsonify({
            "summary": summary
        })


    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "error": "Summary generation failed.",
            "details": str(e)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    print()
    print("=" * 60)
    print("ParaBrief AI server started")
    print("URL: http://127.0.0.1:" + str(port))
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
