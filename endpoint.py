import time
import uvicorn
from typing import Dict
from fastapi import FastAPI
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

def fix_tokenizer(tokenizer, new_lang='quz_Latn'):
    """
    Add a new language token to the tokenizer vocabulary and update language mappings.
    """
    # First ensure we're working with an NLLB tokenizer
    if not hasattr(tokenizer, 'sp_model'):
        raise ValueError("This function expects an NLLB tokenizer")

    # Add the new language token if it's not already present
    if new_lang not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({
            'additional_special_tokens': [new_lang]
        })
    
    # Initialize lang_code_to_id if it doesn't exist
    if not hasattr(tokenizer, 'lang_code_to_id'):
        tokenizer.lang_code_to_id = {}
        
    # Add the new language to lang_code_to_id mapping
    if new_lang not in tokenizer.lang_code_to_id:
        # Get the ID for the new language token
        new_lang_id = tokenizer.convert_tokens_to_ids(new_lang)
        tokenizer.lang_code_to_id[new_lang] = new_lang_id
        
    # Initialize id_to_lang_code if it doesn't exist
    if not hasattr(tokenizer, 'id_to_lang_code'):
        tokenizer.id_to_lang_code = {}
        
    # Update the reverse mapping
    tokenizer.id_to_lang_code[tokenizer.lang_code_to_id[new_lang]] = new_lang

    return tokenizer


MODEL_URL = "pollitoconpapass/QnIA-translation-model"
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_URL)
tokenizer = NllbTokenizer.from_pretrained(MODEL_URL)
fix_tokenizer(tokenizer)


# === TRANSLATION ===
@app.post("/qnia-translate")
async def translate(data: Dict, a=32, b=3, max_input_length=1024, num_beams=4):
    start = time.time()

    text = data['text']
    src_lang = data['src_lang']
    tgt_lang = data['tgt_lang']

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length)
    
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams,
    )

    translation = tokenizer.batch_decode(result, skip_special_tokens=True)
    translation = translation[0]

    end = time.time()
    print(f"\nTime: {end - start}")

    return {'translation': translation}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8889)
