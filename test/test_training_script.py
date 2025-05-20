import re
import os
import gc
import sys
import torch
import random
import datasets
import unicodedata
import numpy as np
import typing as tp
import pandas as pd
import plotly.graph_objects as go
from tqdm.auto import tqdm, trange
from sacremoses import MosesPunctNormalizer
from transformers.optimization import Adafactor
from torch.cuda.amp import autocast, GradScaler
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, get_constant_schedule_with_warmup

# ===========================
# === DATASET PREPARATION ===
# ===========================
DATASET__NAME = 'pollitoconpapass/new-cuzco-quechua-translation-dataset'
TRAINING_COLUMN_NAME = "train"
VALIDATION_COLUMN_NAME = "eval"

SRC_LANGUAGE_COLUMN_NAME = "spanish" 
TGT_LANGUAGE_COLUMN_NAME = "quechua"

lang_dataset = datasets.load_dataset(DATASET__NAME)
print(f"Example: \n{lang_dataset[TRAINING_COLUMN_NAME][22]}")

# Training dataset
training_tgt = []
training_src = []

for i in range(len(lang_dataset[TRAINING_COLUMN_NAME])):
  training_tgt.append(lang_dataset[TRAINING_COLUMN_NAME][i][TGT_LANGUAGE_COLUMN_NAME])
  training_src.append(lang_dataset[TRAINING_COLUMN_NAME][i][SRC_LANGUAGE_COLUMN_NAME])

# Validation dataset
validation_tgt = []
validation_src = []

for i in range(len(lang_dataset[VALIDATION_COLUMN_NAME])):
  validation_tgt.append(lang_dataset[VALIDATION_COLUMN_NAME][i][TGT_LANGUAGE_COLUMN_NAME])
  validation_src.append(lang_dataset[VALIDATION_COLUMN_NAME][i][SRC_LANGUAGE_COLUMN_NAME])

df_train = pd.DataFrame({"spa": training_src, "quz": training_tgt}) # 619 (big) items
df_dev = pd.DataFrame({"spa": validation_src, "quz": validation_tgt})   # 16k items

print(df_train.shape)
print(df_dev.shape)


# ===============================
# === NLLB TOKENIZER HANDLING ===
# ===============================
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = NllbTokenizer.from_pretrained(NLLB_MODEL_NAME)

def word_tokenize(text):
    return re.findall('(\w+|[^\w\s])', text)

smpl = df_train.sample(100, random_state=1)
print(smpl)
print(smpl.sample(5))

stats = smpl[['spa', 'quz']].applymap(len).describe()
print(stats)

print(stats.spa['mean'] / stats.quz['mean'])
print(tokenizer.unk_token, tokenizer.unk_token_id)

texts_with_unk = [text for text in tqdm(df_train.quz) if tokenizer.unk_token_id in tokenizer(text).input_ids]
print(len(texts_with_unk))

s = random.sample(texts_with_unk, 5)
print(s)

mpn = MosesPunctNormalizer(lang="es") # -> change this depending on your src language
mpn.substitutions = [
    (re.compile(r), sub) for r, sub in mpn.substitutions
]

def get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char

replace_nonprint = get_non_printing_char_replacer(" ")

def preproc(text):
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
    clean = unicodedata.normalize("NFKC", clean)
    return clean


# =============================================
# === ADDING NEW LANGUAGE TO NLLB TOKENIZER ===
# =============================================
NEW_NLLB_LANG_CODE = "quz_Latn"
SRC_NLLB_LANG_CODE = "spa_Latn"
SIMILAR_NLLB_LANG_CODE = "quy_Latn" # -> you'll need a similar language to the one you want to finetune on (in my case: Ayacucho Quechua)

tokenizer = NllbTokenizer.from_pretrained(NLLB_MODEL_NAME)
print(len(tokenizer))
print(tokenizer.convert_ids_to_tokens([256202, 256203]))

def fix_tokenizer(tokenizer, new_lang=NEW_NLLB_LANG_CODE):
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

fix_tokenizer(tokenizer)

# Verify the mappings are correct
print(f"Language ID mapping: {tokenizer.lang_code_to_id.get(NEW_NLLB_LANG_CODE)}")
print(f"Is language in special tokens: {(NEW_NLLB_LANG_CODE in tokenizer.additional_special_tokens)}")

# If you need to support the language as both source and target,
# make sure it's added to both sets if NLLB uses these
if hasattr(tokenizer, 'src_lang_codes'):
    if NEW_NLLB_LANG_CODE not in tokenizer.src_lang_codes:
        tokenizer.src_lang_codes.append(NEW_NLLB_LANG_CODE)
        
if hasattr(tokenizer, 'tgt_lang_codes'):
    if NEW_NLLB_LANG_CODE not in tokenizer.tgt_lang_codes:
        tokenizer.tgt_lang_codes.append(NEW_NLLB_LANG_CODE)

print(tokenizer.convert_ids_to_tokens([256202, 256203, 256204])) # ['zul_Latn', '<mask>', 'quz_Latn']
print(tokenizer.convert_tokens_to_ids(['zul_Latn', NEW_NLLB_LANG_CODE, '<mask>'])) # [256202, 256204, 256203]

added_token_id = tokenizer.convert_tokens_to_ids(NEW_NLLB_LANG_CODE)
similar_lang_id = tokenizer.convert_tokens_to_ids(SIMILAR_NLLB_LANG_CODE)
print(added_token_id, similar_lang_id)

model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

# moving the embedding for "mask" to its new position
model.model.shared.weight.data[added_token_id + 1] = model.model.shared.weight.data[added_token_id]
# initializing new language token with a token of a similar language
model.model.shared.weight.data[added_token_id] = model.model.shared.weight.data[similar_lang_id]


# =======================================
# === PREPARING FOR THE TRAINING LOOP ===
# =======================================
def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

cleanup()

torch.backends.cudnn.benchmark = True
model.cuda()
model.gradient_checkpointing_enable()

scaler = GradScaler()

optimizer = Adafactor(
    [p for p in model.parameters() if p.requires_grad],
    scale_parameter=False,
    relative_step=False,
    lr=1e-4,
    clip_threshold=1.0,
    weight_decay=1e-3,
)

# Training config
batch_size = 16
max_length = 128
warmup_steps = 1_000
training_steps = 10000 # -> the original code it appears 57000 (it takes 37 hours)
accumulation_steps = 4  # Simulate larger batch size
effective_batch_size = batch_size * accumulation_steps

losses = []
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

LANGS = [('spa', SRC_NLLB_LANG_CODE), ('quz', NLLB_MODEL_NAME)]

def get_batch_pairs(batch_size, data=df_train):
    (l1, long1), (l2, long2) = random.sample(LANGS, 2)
    xx, yy = [], []
    for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data)-1)]
        xx.append(preproc(item[l1]))
        yy.append(preproc(item[l2]))
    return xx, yy, long1, long2

print(get_batch_pairs(1))

@torch.compile  # Only if using PyTorch 2.0+
def tokenize_batch(tokenizer, texts, lang, max_length, device):
    tokenizer.src_lang = lang
    return tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

MODEL_SAVE_PATH = './finetuned_models'

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

def evaluate_model(model, tokenizer, test_data, batch_size=8):
    """
    Evaluate the model's performance on test data
    
    Args:
        model: The NLLB model
        tokenizer: The NLLB tokenizer
        test_data: DataFrame containing test pairs
        batch_size: Batch size for evaluation
    
    Returns:
        float: Average loss on test set
    """
    model.eval()  # Set model to evaluation mode
    eval_losses = []
    
    # Disable gradient calculations for evaluation
    with torch.no_grad():
        # Get batches from test data
        for _ in range(0, len(test_data), batch_size):
            try:
                # Get a batch of test pairs
                xx, yy, lang1, lang2 = get_batch_pairs(batch_size, data=test_data)
                
                # Tokenize input
                tokenizer.src_lang = lang1
                x = tokenizer(xx, return_tensors='pt', padding=True, 
                            truncation=True, max_length=max_length).to(model.device)
                
                # Tokenize target
                tokenizer.src_lang = lang2
                y = tokenizer(yy, return_tensors='pt', padding=True, 
                            truncation=True, max_length=max_length).to(model.device)
                y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100
                
                # Get loss
                outputs = model(**x, labels=y.input_ids)
                eval_losses.append(outputs.loss.item())
                
            except RuntimeError as e:
                print(f"Error in evaluation: {e}")
                continue
    
    model.train()  # Set model back to training mode
    return np.mean(eval_losses)


# =====================
# === TRAINING LOOP ===
# =====================
model.train()
x, y, loss = None, None, None
cleanup()

running_loss = 0.0
tq = trange(len(losses), training_steps)

#trying early stopping...
initial_loss = evaluate_model(model, tokenizer, df_train)
print(f"Initial test loss: {initial_loss}")

best_loss = initial_loss
patience = 5
patience_counter = 0

for i in tq:
    # Zero gradients only once per accumulation cycle
    if i % accumulation_steps == 0:
        optimizer.zero_grad(set_to_none=True)
        
    try:
        xx, yy, lang1, lang2 = get_batch_pairs(batch_size)
        
        # Use mixed precision training
        with autocast():
            # Tokenize inputs
            x = tokenize_batch(tokenizer, xx, lang1, max_length, model.device)
            y = tokenize_batch(tokenizer, yy, lang2, max_length, model.device)
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

            # Forward pass
            loss = model(**x, labels=y.input_ids).loss / accumulation_steps
            
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            
            running_loss += loss.item() * accumulation_steps

        # Update weights if accumulation cycle is complete
        if (i + 1) % accumulation_steps == 0:
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer and scheduler step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Record loss
            losses.append(running_loss)
            running_loss = 0.0
            
            # Update progress bar
            tq.set_description(f"Loss: {np.mean(losses[-100:]):.4f}")

    except RuntimeError as e:
        optimizer.zero_grad(set_to_none=True)
        x, y, loss = None, None, None
        cleanup()
        print('error', max(len(s) for s in xx + yy), e)
        continue

    # Periodic cleanup every 100 steps
    if i % 100 == 0:
        cleanup()

    if i % 1000 == 0:
        print(f"Step {i}, Average Loss: {np.mean(losses[-1000:]):.4f}")

    if i % 1000 == 0 and i > 0:
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)


    if i % 5000 == 0:
        test_loss = evaluate_model(model, tokenizer, df_train)
        print(f"Step {i}, Test Loss: {test_loss}")

        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0

            best_model_path = f"{MODEL_SAVE_PATH}_best"

            if not os.path.exists(best_model_path):
                os.makedirs(best_model_path)

            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
        
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early Stopping Triggered!!")
                break

# Show the loss in a graphic
fig = go.Figure()
fig.add_trace(go.Scatter(y=pd.Series(losses).ewm(100).mean(),
                        mode='lines',
                        name='EWM Loss'))
fig.show()

def translate(text, src_lang=SRC_NLLB_LANG_CODE, tgt_lang=NEW_NLLB_LANG_CODE, a=16, b=1.5, max_input_length=1024, **kwargs):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length)
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        **kwargs
    )
    #print(inputs.input_ids.shape[1], result.shape[1])
    return tokenizer.batch_decode(result, skip_special_tokens=True)

xx, yy, lang1, lang2 = get_batch_pairs(1, data=df_dev)
print(xx)
print(yy)
model.eval()
print(translate(xx[0], lang1, lang2, no_repeat_ngram_size=3, num_beams=5))


# =======================
# === USING THE MODEL ===
# =======================
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

model_load_name = MODEL_SAVE_PATH
model = AutoModelForSeq2SeqLM.from_pretrained(model_load_name).cuda()
tokenizer = NllbTokenizer.from_pretrained(model_load_name)
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
    return tokenizer.batch_decode(result, skip_special_tokens=True)

t = "Hola buenos días mi nombre es Jose"
print(f"First translation test:\n {translate(t, SRC_NLLB_LANG_CODE, NEW_NLLB_LANG_CODE)}")


# ========================
# === PUBLISHING TO HF ===
# ========================
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

model_load_name = MODEL_SAVE_PATH
model = AutoModelForSeq2SeqLM.from_pretrained(model_load_name)
tokenizer = NllbTokenizer.from_pretrained(model_load_name)
fix_tokenizer(tokenizer)

upload_repo = "pollitoconpapass/QnIA-translation-model"
tokenizer.push_to_hub(upload_repo)
model.push_to_hub(upload_repo)

print("Model published to HF! All done now...")
