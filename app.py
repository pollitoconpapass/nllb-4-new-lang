import time
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM


# === GENERAL PREVIOUS CONFIGURATION (change to accomodate your needs) ===
NEW_NLLB_LANG_CODE = 'quz_Latn'
SRC_NLLB_LANG_CODE = 'spa_Latn'
FINETUNED_MODEL = "pollitoconpapass/QnIA-translation-model"

def fix_tokenizer(tokenizer, new_lang=NEW_NLLB_LANG_CODE):
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


model = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED_MODEL)
tokenizer = NllbTokenizer.from_pretrained(FINETUNED_MODEL)
fix_tokenizer(tokenizer)

def translate(text, src_lang=SRC_NLLB_LANG_CODE, tgt_lang=NEW_NLLB_LANG_CODE, a=32, b=3, max_input_length=1024, num_beams=4, **kwargs):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length)
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams,
        **kwargs
    )
    
    translation = tokenizer.batch_decode(result, skip_special_tokens=True)
    translation = translation[0]
    return translation


# === MAIN === (change to accomodate your needs)
t = '''
Buenos días, mi nombre es Jose.
'''

start = time.time()
result_v1 = translate(t, 'spa_Latn', 'quz_Latn')
print(f"\n{result_v1}")

end = time.time()
print(f"\nTime: {end - start}")
