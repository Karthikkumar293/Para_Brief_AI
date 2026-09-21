from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch
import os

app = Flask(__name__, static_folder=".")

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "best_3"

print("=" * 60)
print("Starting ParaBrief AI")
print("=" * 60)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Loading tokenizer...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Tokenizer loaded.")

print("Loading BART summarization model...")

# Load BART model
model = BartForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.to(device)
model.eval()

print("Model loaded successfully.")
print("=" * 60)


# ============================================================
# FRONTEND
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

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received."
            }), 400

        text = data.get("text", "").strip()

        if not text:
            return jsonify({
                "error": "Please enter some text."
            }), 400

        # ----------------------------------------------------
        # TOKENIZATION
        # ----------------------------------------------------

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        # Move tensors to device
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # SUMMARY GENERATION
        # ----------------------------------------------------

        with torch.no_grad():

            output_ids = model.generate(
                **inputs,

                # Generation configuration
                num_beams=4,
                early_stopping=True,

                # Prevent repetitive phrases
                no_repeat_ngram_size=3,

                # Summary length
                min_length=56,
                max_length=142,

                # Generation tokens
                decoder_start_token_id=model.config.decoder_start_token_id,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id
            )

        # ----------------------------------------------------
        # DETOKENIZATION
        # ----------------------------------------------------

        summary = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        summary = summary.strip()

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({
            "summary": summary
        })

    except Exception as e:

        print("Summarization error:")
        print(str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model": "ParaBrief AI",
        "architecture": "BART Encoder-Decoder"
    })


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("ParaBrief AI is running")
    print("Open: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
