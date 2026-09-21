from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch
import os

app = Flask(__name__, static_folder=".")

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "best_3"

print("=" * 60)
print("Starting ParaBrief AI")
print("=" * 60)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Loading tokenizer...")

# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Tokenizer loaded.")

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading BART summarization model...")

model = BartForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    local_files_only=True
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
    return send_from_directory(
        os.path.abspath("."),
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model": "ParaBrief AI",
        "architecture": "BART Encoder-Decoder",
        "device": str(device)
    })


# ============================================================
# SUMMARIZATION API
# ============================================================

@app.route("/summarize", methods=["POST"])
def summarize():

    try:

        # ----------------------------------------------------
        # GET JSON DATA
        # ----------------------------------------------------

        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "error": "No JSON data received."
            }), 400

        text = data.get("text", "")

        if not isinstance(text, str):
            return jsonify({
                "error": "Text must be a string."
            }), 400

        text = text.strip()

        if not text:
            return jsonify({
                "error": "Please enter some text."
            }), 400

        print()
        print("=" * 60)
        print("SUMMARIZATION REQUEST")
        print("=" * 60)
        print("Input length:", len(text))
        print("Input text:", text[:500])

        # ----------------------------------------------------
        # TOKENIZE
        # ----------------------------------------------------

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )

        # Move tensors to CPU/GPU
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        print("Input tokens:", inputs["input_ids"].shape)

        # ----------------------------------------------------
        # GENERATE SUMMARY
        # ----------------------------------------------------

        with torch.no_grad():

            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],

                num_beams=4,

                min_length=20,
                max_length=100,

                no_repeat_ngram_size=3,

                length_penalty=2.0,

                early_stopping=True,

                decoder_start_token_id=(
                    model.config.decoder_start_token_id
                ),

                eos_token_id=model.config.eos_token_id,

                pad_token_id=model.config.pad_token_id
            )

        # ----------------------------------------------------
        # DECODE SUMMARY
        # ----------------------------------------------------

        summary = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        summary = summary.strip()

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if not summary:

            return jsonify({
                "error": "The model did not generate a summary."
            }), 500

        print("Generated summary:")
        print(summary)

        print("=" * 60)

        # ----------------------------------------------------
        # RETURN JSON
        # ----------------------------------------------------

        return jsonify({
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary)
        })

    except Exception as e:

        print()
        print("=" * 60)
        print("SUMMARIZATION ERROR")
        print("=" * 60)
        print(repr(e))
        print("=" * 60)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("ParaBrief AI is running")
    print("Open: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )

