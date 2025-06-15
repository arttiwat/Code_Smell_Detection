import os
import boto3
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
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
        "s3_prefix": "data class/CodeT5_Data_Class",  # folder in S3
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
        "s3_prefix": "long method/Cbest_CodeT5_Long_Method_batch_size_64_seq_len_512_lr_1e-05_scheduler_linear_chunk_static_logic_name_combined_Time_20250221-041313",
        "local_dir": "models/long_method",
        "tokenizer_name": "Salesforce/codet5-base",
    },
}

MAX_TOKENS = 512

# --------------------
# Download model folder from S3 if not already downloaded
def download_model_folder_from_s3(bucket: str, prefix: str, local_dir: str):
    if Path(local_dir).exists():
        st.write(f"✔️ Model already cached at `{local_dir}`.")
        return

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    st.write(f"⬇ Downloading model files from `s3://{bucket}/{prefix}`...")
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            rel_path = s3_key[len(prefix):].lstrip("/")
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, s3_key, local_path)
            st.write(f"  ✅ {rel_path}")

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
def chunk_text(tokenizer, text: str, max_tokens: int) -> List[List[int]]:
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]

# --------------------
# Predict on chunk
def predict_chunk(model, tokenizer, chunk_tokens: List[int]) -> int:
    import torch.nn.functional as F
    input_ids = torch.tensor([chunk_tokens])
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        return torch.argmax(probs, dim=1).item()

# --------------------
# Predict full text
def predict_smell_for_text(model, tokenizer, text: str) -> str:
    chunks = chunk_text(tokenizer, text, MAX_TOKENS)
    for chunk in chunks:
        if predict_chunk(model, tokenizer, chunk) == 1:
            return "Smell"
    return "Non-smell"

# --------------------
# Streamlit UI
def main():
    st.set_page_config(page_title="Code Smell Detector", layout="wide")
    st.title("🧠 Code Smell Detector")
    st.write("Paste your code below to detect code smells using CodeBERT and CodeT5.")

    models, tokenizers = load_models_and_tokenizers()

    code_input = st.text_area("Paste your code here:", height=300)

    if st.button("🔍 Detect Code Smells"):
        if not code_input.strip():
            st.warning("⚠️ Please enter some code to analyze.")
            return

        st.write("🔎 Analyzing...")
        results = {}
        for smell, model in models.items():
            tokenizer = tokenizers[smell]
            result = predict_smell_for_text(model, tokenizer, code_input)
            results[smell] = result

        st.subheader("📊 Prediction Results")
        for smell, result in results.items():
            emoji = "⚠️" if result == "Smell" else "✅"
            st.write(f"{emoji} **{smell}:** {result}")

if __name__ == "__main__":
    main()
