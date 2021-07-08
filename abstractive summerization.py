# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 01:18:09 2021

@author: anirb
"""


"""
https://huggingface.co/transformers/model_doc/pegasus.html#transformers.TFPegasusModel
https://huggingface.co/google/pegasus-large?text=Coronavirus+disease+%28COVID-19%29+is+an+infectious+disease+caused+by+a+newly+discovered+coronavirus.%5C%0AMost+people+infected+with+the+COVID-19+virus+will+experience+mild+to+moderate+respiratory+illness+and+%5C%0Arecover+without+requiring+special+treatment.++Older+people%2C+and+those+with+underlying+medical+problems+%5C%0Alike+cardiovascular+disease%2C+diabetes%2C+chronic+respiratory+disease%2C+and+cancer+are+more+likely+to+develop+serious+illness.%5C%0AThe+best+way+to+prevent+and+slow+down+transmission+is+to+be+well+informed+about+the+COVID-19+virus%2C+%5C%0Athe+disease+it+causes+and+how+it+spreads.+Protect+yourself+and+others+from+infection+by+washing+your+%5C%0Ahands+or+using+an+alcohol+based+rub+frequently+and+not+touching+your+face.+%0AThe+COVID-19+virus+spreads+primarily+through+droplets+of+saliva+or+discharge+%5C%0Afrom+the+nose+when+an+infected+person+coughs+or+sneezes%2C+so+it%27s+important+that+%5C%0Ayou+also+practice+respiratory+etiquette+%28for+example%2C+by+coughing+into+a+flexed+elbow%29.
https://towardsdatascience.com/how-to-perform-abstractive-summarization-with-pegasus-3dd74e48bafb#:~:text=On%20a%20high%20level%2C%20PEGASUS,representation%20of%20the%20input%20text.


https://towardsdatascience.com/simple-abstractive-text-summarization-with-pretrained-t5-text-to-text-transfer-transformer-10f6d602c426
"""



from transformers import PegasusForConditionalGeneration, PegasusTokenizer,\
    TFPegasusForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration, T5Config

import torch
import tensorflow as tf


src_text = ["""Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus.\ 
Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and \
recover without requiring special treatment.  Older people, and those with underlying medical problems \
like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.\ 
The best way to prevent and slow down transmission is to be well informed about the COVID-19 virus, \
the disease it causes and how it spreads. Protect yourself and others from infection by washing your \
hands or using an alcohol based rub frequently and not touching your face. \
The COVID-19 virus spreads primarily through droplets of saliva or discharge \
from the nose when an infected person coughs or sneezes, so it's important that \
you also practice respiratory etiquette (for example, by coughing into a flexed elbow)."""]


# model_name = 'google/pegasus-xsum'
model_name = 'google/pegasus-large'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = PegasusTokenizer.from_pretrained(model_name)


# model = TFPegasusForConditionalGeneration.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(model_name)
# inputs = tokenizer(src_text, max_length=1024, return_tensors='tf',\
#                    truncation=True, padding='longest')

inputs = tokenizer(src_text, return_tensors='tf',\
                   truncation=True, padding='longest')




# Generate Summary
summary_ids = model.generate(inputs['input_ids'])
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

