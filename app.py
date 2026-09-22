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
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

        # Move tensors to CPU/GPU
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        input_token_count = inputs["input_ids"].shape[1]

        print("Input tokens:", input_token_count)


        # ----------------------------------------------------
        # CALCULATE SUMMARY LENGTH
        # ----------------------------------------------------

        # Summary should normally be shorter than the input.
        max_summary_length = min(
            142,
            max(30, int(input_token_count * 0.45))
        )

        min_summary_length = min(
            30,
            max(10, int(input_token_count * 0.15))
        )

        # Make sure min < max
        if min_summary_length >= max_summary_length:
            min_summary_length = max(
                5,
                max_summary_length // 2
            )

        print(
            "Summary length:",
            min_summary_length,
            "-",
            max_summary_length
        )


        # ----------------------------------------------------
        # GENERATE SUMMARY
        # ----------------------------------------------------

        print("Generating summary...")

        with torch.no_grad():

            output_ids = model.generate(

                input_ids=inputs["input_ids"],

                attention_mask=inputs["attention_mask"],


                # Beam search
                num_beams=6,


                # Encourage concise summaries
                length_penalty=2.0,


                # Reduce repetition
                no_repeat_ngram_size=3,

                repetition_penalty=1.2,


                # Dynamic summary length
                min_length=min_summary_length,

                max_length=max_summary_length,


                # Stop when EOS is generated
                early_stopping=True,


                # BART generation tokens
                decoder_start_token_id=(
                    model.config.decoder_start_token_id
                ),

                eos_token_id=(
                    model.config.eos_token_id
                ),

                pad_token_id=(
                    model.config.pad_token_id
                ),

                # Fix BART configuration warning
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
        # CHECK EMPTY OUTPUT
        # ----------------------------------------------------

        if not summary:

            return jsonify({
                "error": "The model did not generate a summary."
            }), 500


        # ----------------------------------------------------
        # CHECK IF MODEL COPIED INPUT
        # ----------------------------------------------------

        normalized_input = " ".join(
            text.lower().split()
        )

        normalized_summary = " ".join(
            summary.lower().split()
        )


        if normalized_input == normalized_summary:

            print()
            print("WARNING:")
            print("Model returned the input unchanged.")
            print("Trying alternative generation...")


            # ------------------------------------------------
            # ALTERNATIVE GENERATION
            # ------------------------------------------------

            alternative_max_length = min(
                100,
                max(30, int(input_token_count * 0.35))
            )

            alternative_min_length = min(
                20,
                max(8, int(input_token_count * 0.10))
            )

            if alternative_min_length >= alternative_max_length:
                alternative_min_length = max(
                    5,
                    alternative_max_length // 2
                )


            with torch.no_grad():

                output_ids = model.generate(

                    input_ids=inputs["input_ids"],

                    attention_mask=inputs["attention_mask"],


                    # Stronger beam search
                    num_beams=8,


                    # More concise output
                    length_penalty=2.5,


                    # Repetition control
                    no_repeat_ngram_size=3,

                    repetition_penalty=1.3,


                    min_length=alternative_min_length,

                    max_length=alternative_max_length,


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


            # Decode alternative result
            summary = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            summary = summary.strip()


        # ----------------------------------------------------
        # FINAL EMPTY CHECK
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

        print("=" * 60)


        # ----------------------------------------------------
        # RETURN JSON
        # ----------------------------------------------------

        return jsonify({
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary)
        })


    # ========================================================
    # ERROR HANDLING
    # ========================================================

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
