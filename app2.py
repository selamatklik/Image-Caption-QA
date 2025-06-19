import streamlit as st
import os, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from torchvision import transforms
from io import BytesIO
from pathlib import Path
import pandas as pd

torch.classes.__path__ = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load dataframe
path = "C:/lio/AI for Enterprise/cxr-dataset"
df_dir = path + "/Cleanses csv tfrecords/"
df_val_name = "df_val.csv"
mimic_df_test = pd.read_csv(Path(df_dir + df_val_name))

# Prepare Dataset
test_img_dir = path + '/mimic_dset/re_512_3ch/Test'
mimic_df_test['path'] = mimic_df_test.apply(lambda x: f"{test_img_dir}/{x['id']}.jpg", axis=1)
mimic_df_test = mimic_df_test[:10]

def load_images(path):
  img = Image.open(path)
  img = img.convert('RGB')
  return img

@st.cache_resource
def load_caption_model():   
    # load medicap
    ckpt_name = 'aehrc/medicap'
    
    local_folder = "model/"
    if os.path.exists(local_folder):
        medicap = transformers.AutoModel.from_pretrained(local_folder, trust_remote_code=True)
    else:
        medicap = transformers.AutoModel.from_pretrained(ckpt_name, trust_remote_code=True)
    medicap = medicap.to(device)
    medicap.eval()

    # transform image 
    medicap_transforms = transformers.AutoFeatureExtractor.from_pretrained(ckpt_name)

    # tokenizer
    medicap_tokenizer = transformers.GPT2Tokenizer.from_pretrained(ckpt_name)

    return medicap, medicap_transforms, medicap_tokenizer

def generate_image_caption(image, model, transformer, tokenizer):
    image = transformer(image, return_tensors="pt")
    image = image["pixel_values"]
    outputs = model.generate(
        pixel_values=image.to(device),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_length=128,
        num_beams=4,
        output_attentions=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@st.cache_resource
def load_qa_model():
    model_name = "microsoft/BioGPT-Large-PubMedQA"

    local_folder = "BioGPT-Large-PubMedQA/"
    if os.path.exists(local_folder):
        biogpt_tokenizer = AutoTokenizer.from_pretrained(local_folder)
        biogpt = AutoModelForCausalLM.from_pretrained(local_folder)
    else:
        biogpt_tokenizer = AutoTokenizer.from_pretrained(model_name)
        biogpt = AutoModelForCausalLM.from_pretrained(model_name)
    biogpt = biogpt.to(device)
    biogpt.eval()

    return biogpt, biogpt_tokenizer

def generate_answer(description, question, model, tokenizer):
    prompt = f"question: {question} context: {description}"
    new_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    input_ids = new_input_ids

    generated_output = model.generate(
        input_ids,
        max_new_tokens=128,  # Max new tokens for the bot's response
    )

    response = tokenizer.decode(generated_output[0], skip_special_tokens=True)

    return response

st.set_page_config(page_title="Image Caption + QA", layout="centered")
st.title("🖼️ Caption-Based Question Answering")

# Dropdown list
options = range(10)
choice = st.selectbox("Choose an action:", options)
if choice is not None:
    data = mimic_df_test.iloc[choice]
    label = data['text']
    img = Image.open(Path(data['path']))
    st.image(img)
    st.subheader("📝 Original Description")
    st.info(label)

    # image description
    medicap, medicap_transforms, medicap_tokenizer = load_caption_model()
    caption = generate_image_caption(img, medicap, medicap_transforms, medicap_tokenizer)

    st.subheader("📝 Generated Description")
    st.info(caption)

    # vqa
    st.markdown("---")

    st.subheader("❓ Ask a Question About the Image")
    question = st.text_input("Type your question")

    if question:
        biogpt, biogpt_tokenizer = load_qa_model()
        response = generate_answer(caption, question, biogpt, biogpt_tokenizer)
        st.success(f"{response}")

else:
    st.info("Please upload an image file.")