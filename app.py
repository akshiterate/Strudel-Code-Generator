from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Log all incoming requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Check if static directory exists
STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    raise RuntimeError(f"Static directory '{STATIC_DIR}' does not exist. Please create it and add HTML files.")

# Check if model and vocabulary files exist
MODEL_FILES = {
    "lstm": "LSTM_model.keras",
    "bilstm": "BiLSTM_model.keras",
    "transformer": "transformer_model.keras"
}
DATA_DIR = "processed_data"
VOCAB_FILES = ["vocabulary.json", "reverse_vocab.json"]

for model_name, model_path in MODEL_FILES.items():
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file '{model_path}' does not exist.")
for vocab_file in VOCAB_FILES:
    if not os.path.exists(os.path.join(DATA_DIR, vocab_file)):
        raise RuntimeError(f"Vocabulary file '{os.path.join(DATA_DIR, vocab_file)}' does not exist.")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Allow CORS for browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set mixed precision policy for Transformer model
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Define custom layers for Transformer model
@keras.utils.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, seq_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = seq_length
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(seq_length, d_model)
    
    def build(self, input_shape):
        pass  # Added to suppress warning
    
    def get_config(self):
        config = super().get_config()
        config.update({"seq_length": self.seq_length, "d_model": self.d_model})
        return config
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def positional_encoding(self, seq_length, d_model):
        angle_rads = self.get_angles(
            np.arange(seq_length)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float16)
    
    def call(self, x):
        pos_enc = tf.cast(self.pos_encoding[:, :tf.shape(x)[1], :], dtype=x.dtype)
        return x + pos_enc

@keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu", dtype='float16'),
            layers.Dense(embed_dim, dtype='float16'),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, dtype='float16')
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, dtype='float16')
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def build(self, input_shape):
        pass  # Added to suppress warning
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config
    
    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float16)
        attn_output = self.att(inputs, inputs, attention_mask=self.get_causal_mask(tf.shape(inputs)[1]), training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_causal_mask(self, seq_length):
        mask = tf.linalg.band_part(tf.ones((seq_length, seq_length), dtype=tf.float16), -1, 0)
        return mask

# Load vocabularies
SEQUENCE_LENGTH = 50
with open(f"{DATA_DIR}/vocabulary.json", "r") as f:
    vocabulary = json.load(f)
with open(f"{DATA_DIR}/reverse_vocab.json", "r") as f:
    reverse_vocab = {int(k): v for k, v in json.load(f).items()}

# Load models
try:
    lstm_model = keras.models.load_model("LSTM_model.keras")
    bilstm_model = keras.models.load_model("BiLSTM_model.keras")
    transformer_model = keras.models.load_model(
        "transformer_model.keras",
        custom_objects={"PositionalEncoding": PositionalEncoding, "TransformerBlock": TransformerBlock}
    )
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

def generate_strudel_code(model, seed_text=None, max_tokens=200, temperature=1.0, num_lines=4):
    if seed_text is None:
        generated_tokens = [vocabulary["<START>"], vocabulary["<CPM:120/4>"]]
    else:
        generated_tokens = [vocabulary.get(token, vocabulary["<PAD>"]) for token in seed_text.split()]

    for i in range(max_tokens):
        if len(generated_tokens) < SEQUENCE_LENGTH:
            padded = [vocabulary["<PAD>"]] * (SEQUENCE_LENGTH - len(generated_tokens))
            input_seq = padded + generated_tokens
        else:
            input_seq = generated_tokens[-SEQUENCE_LENGTH:]

        input_seq = np.array([input_seq])
        predictions = model.predict(input_seq, verbose=0)
        next_token_probs = predictions[0, -1, :]
        next_token_idx = sample_with_temperature(next_token_probs, temperature)
        generated_tokens.append(next_token_idx)

        if reverse_vocab[next_token_idx] == "<SEP>":
            sep_count = sum(1 for t in generated_tokens if reverse_vocab[t] == "<SEP>")
            if sep_count >= num_lines:
                break

    generated_text_tokens = [reverse_vocab.get(idx, "<UNK>") for idx in generated_tokens]
    return format_as_strudel(generated_text_tokens)

def format_as_strudel(tokens):
    output = []
    current_line = []
    cpm_value = "120/4"

    for token in tokens:
        if token.startswith("<CPM:"):
            cpm_value = token.replace("<CPM:", "").replace(">", "")
        elif token == "<START>":
            continue
        elif token == "<SEP>":
            if current_line:
                output.append(" ".join(current_line))
                current_line = []
            output.append(">`)")
            output.append("$: note(`<")
        elif token == "<END>":
            if current_line:
                output.append(" ".join(current_line))
                current_line = []
        elif token not in ["<PAD>", "<UNK>"]:
            current_line.append(token)

    if current_line:
        output.append(" ".join(current_line))

    code = f"setcpm({cpm_value})\n"
    code += "$: note(`<\n"
    code += "\n".join(output)
    if not output[-1].endswith(">`)"):
        code += "\n>`)"

    # Remove empty note sections
    lines = code.split("\n")
    cleaned_lines = []
    skip_next = False
    for i, line in enumerate(lines):
        if line.strip() == "$: note(`<":
            if i + 1 < len(lines) and lines[i + 1].strip() == ">`)":
                skip_next = True
                continue
            else:
                cleaned_lines.append(line)
        elif line.strip() == ">`)":
            if not skip_next:
                cleaned_lines.append(line)
            skip_next = False
        else:
            cleaned_lines.append(line)
            skip_next = False
    return "\n".join(cleaned_lines).strip()

class GenerateRequest(BaseModel):
    temperature: float = 1.5
    max_tokens: int = 150
    num_lines: int = 3
    seed_text: str | None = None

@app.post("/generate/lstm")
async def generate_lstm(request: GenerateRequest, http_request: Request):
    logger.info(f"Received request to /generate/lstm: {request.dict()}")
    try:
        code = generate_strudel_code(
            lstm_model,
            seed_text=request.seed_text,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            num_lines=request.num_lines
        )
        return JSONResponse(content={"code": code})
    except Exception as e:
        logger.error(f"Error in /generate/lstm: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/generate/bilstm")
async def generate_bilstm(request: GenerateRequest, http_request: Request):
    logger.info(f"Received request to /generate/bilstm: {request.dict()}")
    try:
        code = generate_strudel_code(
            bilstm_model,
            seed_text=request.seed_text,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            num_lines=request.num_lines
        )
        return JSONResponse(content={"code": code})
    except Exception as e:
        logger.error(f"Error in /generate/bilstm: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/generate/transformer")
async def generate_transformer(request: GenerateRequest, http_request: Request):
    logger.info(f"Received request to /generate/transformer: {request.dict()}")
    try:
        code = generate_strudel_code(
            transformer_model,
            seed_text=request.seed_text,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            num_lines=request.num_lines
        )
        return JSONResponse(content={"code": code})
    except Exception as e:
        logger.error(f"Error in /generate/transformer: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Strudel Code Generator API"}