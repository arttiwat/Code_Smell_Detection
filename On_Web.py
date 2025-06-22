import os
import boto3
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
from pathlib import Path

# --------------------
# Config
BUCKET_NAME = "weightmodel"  # No "s3://"

# Load secrets from Streamlit's secrets.toml
aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
region = st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")  # or your region

# Create boto3 S3 client using credentials
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region
)

MODEL_INFO = {
    "Data Class": {
        "s3_prefix": "data class/CodeT5_Data_Class",
        "local_dir": "models/data_class",
        "tokenizer_name": "Salesforce/codet5-base",
    },
    "God Class": {
        "s3_prefix": "god class/best_CodeT5_God_Class_batch_size_64_seq_len_512_lr_1e-05_scheduler_linear_chunk_static_logic_name_combined_Time_20250205-054338",
        "local_dir": "models/god_class",
        "tokenizer_name": "Salesforce/codet5-base",
    },
    "Feature Envy": {
        "s3_prefix": "feature envy/best_CodeBERT_Feature_Envy_batch_size_64_seq_len_512_lr_1e-05_scheduler_linear_chunk_static_logic_name_combined_Time_20250225-111507",
        "local_dir": "models/feature_envy",
        "tokenizer_name": "microsoft/codebert-base",
    },
    "Long Method": {
        "s3_prefix": "long method/best_CodeT5_Long_Method_batch_size_64_seq_len_512_lr_1e-05_scheduler_linear_chunk_static_logic_name_combined_Time_20250221-041313",
        "local_dir": "models/long_method",
        "tokenizer_name": "Salesforce/codet5-base",
    },
}

MAX_TOKENS = 512

# --------------------
# Download model folder from S3 if not already downloaded
def download_model_folder_from_s3(bucket: str, prefix: str, local_dir: str):
    if Path(local_dir).exists():
        return

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            rel_path = s3_key[len(prefix):].lstrip("/")
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, s3_key, local_path)

# --------------------
# Load models and tokenizers
@st.cache_resource(show_spinner=False)
def load_models_and_tokenizers():
    models = {}
    tokenizers = {}
    for smell, info in MODEL_INFO.items():
        download_model_folder_from_s3(BUCKET_NAME, info["s3_prefix"], info["local_dir"])
        tokenizer = AutoTokenizer.from_pretrained(info["local_dir"])
        model = AutoModelForSequenceClassification.from_pretrained(info["local_dir"]).eval()
        models[smell] = model
        tokenizers[smell] = tokenizer
    return models, tokenizers

# --------------------
# Chunking
def chunk_text(tokenizer, text: str, max_tokens: int) -> Tuple[List[str], int]:
    tokens = tokenizer.encode(text, add_special_tokens=True)
    total_tokens = len(tokens)
    chunks = [tokens[i:i + max_tokens] for i in range(0, total_tokens, max_tokens)]
    decoded_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    return decoded_chunks, total_tokens

# --------------------
# Predict on chunk
def predict_chunk(model, tokenizer, chunk_text: str) -> int:
    import torch.nn.functional as F

    inputs = tokenizer(
        chunk_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKENS,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        return torch.argmax(probs, dim=1).item()

# --------------------
# Predict full text
def predict_smell_for_text(model, tokenizer, text: str) -> Tuple[str, int, int]:
    chunks, total_tokens = chunk_text(tokenizer, text, MAX_TOKENS)
    for chunk in chunks:
        if predict_chunk(model, tokenizer, chunk) == 1:
            return "Smell", len(chunks), total_tokens
    return "Non-smell", len(chunks), total_tokens

# --------------------
# Streamlit UI
def main():
    st.set_page_config(page_title="Code Smell Detector", layout="wide")
    st.title("🧠 Code Smell Detector")
    st.write("Paste your code below to detect 4 type of code smells: Data Class, God Class, Feature Envy & Long Method.")

    models, tokenizers = load_models_and_tokenizers()

    code_input = st.text_area("Paste your code here:", height=300)

    if st.button("🔍 Detect Code Smells"):
        if not code_input.strip():
            st.warning("⚠️ Please enter some code to analyze.")
            return

        st.write("🔎 Analyzing...")
        results = {}
        chunk_info = None

        for smell, model in models.items():
            tokenizer = tokenizers[smell]
            result, total_chunks, total_tokens = predict_smell_for_text(model, tokenizer, code_input)
            results[smell] = result

            if chunk_info is None:
                chunk_info = (total_chunks, total_tokens)

        # Show Input Info
        st.subheader("📄 Input Data Information")
        st.write(f"**Total Chunks:** {chunk_info[0]}")
        st.write(f"**Total Tokens:** {chunk_info[1]}")

        # Show Prediction Results
        st.subheader("📊 Prediction Results")
        for smell, result in results.items():
            if result == "Smell":
                color = "red"
                emoji = "⚠️"
            else:
                color = "green"
                emoji = "✅"

            st.markdown(f"{emoji} **{smell}:** <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
