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

# Select CPU or GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Tokenizer loaded.")

# ============================================================
# LOAD BART MODEL
# ============================================================

print("Loading BART summarization model...")

model = BartForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.to(device)
model.eval()

# ============================================================
# GENERATION CONFIGURATION
# ============================================================

model.generation_config.forced_bos_token_id = 0

print("Model loaded successfully.")
print("=" * 60)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
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

        # ----------------------------------------------------
        # PRINT REQUEST
        # ----------------------------------------------------

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

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        input_token_count = inputs["input_ids"].shape[1]

        print("Input tokens:", input_token_count)

        # ----------------------------------------------------
        # GENERATE SUMMARY
        # ----------------------------------------------------

        with torch.no_grad():

            output_ids = model.generate(

                input_ids=inputs["input_ids"],

                attention_mask=inputs["attention_mask"],

                # Stronger beam search
                num_beams=8,

                # Encourage concise summaries
                length_penalty=3.0,

                # Prevent repeated phrases
                no_repeat_ngram_size=3,

                repetition_penalty=1.3,

                # Summary size
                min_new_tokens=8,

                max_new_tokens=60,

                # Stop when EOS is generated
                early_stopping=True,

                # BART tokens
                decoder_start_token_id=(
                    model.config.decoder_start_token_id
                ),

                eos_token_id=(
                    model.config.eos_token_id
                ),

                pad_token_id=(
                    model.config.pad_token_id
                ),

                forced_bos_token_id=0
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
        # CHECK EMPTY RESULT
        # ----------------------------------------------------

        if not summary:

            print("ERROR: Empty summary generated.")

            return jsonify({
                "error": "The model did not generate a summary."
            }), 500

        # ----------------------------------------------------
        # CHECK WHETHER MODEL COPIED INPUT
        # ----------------------------------------------------

        normalized_input = " ".join(
            text.lower().split()
        )

        normalized_summary = " ".join(
            summary.lower().split()
        )

        if normalized_input == normalized_summary:

            print("WARNING: Model copied the input.")
            print("Trying second generation strategy...")

            with torch.no_grad():

                output_ids = model.generate(

                    input_ids=inputs["input_ids"],

                    attention_mask=inputs["attention_mask"],

                    num_beams=10,

                    length_penalty=4.0,

                    no_repeat_ngram_size=3,

                    repetition_penalty=1.5,

                    min_new_tokens=6,

                    max_new_tokens=45,

                    early_stopping=True,

                    decoder_start_token_id=(
                        model.config.decoder_start_token_id
                    ),

                    eos_token_id=(
                        model.config.eos_token_id
                    ),

                    pad_token_id=(
                        model.config.pad_token_id
                    ),

                    forced_bos_token_id=0
                )

            summary = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            summary = summary.strip()

        # ----------------------------------------------------
        # FINAL CHECK
        # ----------------------------------------------------

        if not summary:

            return jsonify({
                "error": "The model generated an empty summary."
            }), 500

        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print()
        print("GENERATED SUMMARY:")
        print(summary)

        print()
        print("Original characters:", len(text))
        print("Summary characters:", len(summary))

        print("=" * 60)

        # ----------------------------------------------------
        # RETURN JSON
        # ----------------------------------------------------

        return jsonify({

            "summary": summary,

            "original_length": len(text),

            "summary_length": len(summary)

        })

    # --------------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------------

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
