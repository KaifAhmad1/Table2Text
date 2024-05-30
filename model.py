# model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import dspy
from dsp.modules.lm import LM
from config import MODEL_ID, MAX_OUTPUT_TOKENS
from utils import dataframe_to_string

class HFModel(LM):
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model)
        self.model = model
        self.tokenizer = tokenizer
        self.drop_prompt_from_output = True
        self.history = []
        self.is_client = False
        self.device = model.device
        self.kwargs = {
            "temperature": 0.1,
            "max_new_tokens": 100,
        }

    def basic_request(self, prompt, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        response = self._generate(prompt, **kwargs)
        self.history.append({"prompt": prompt, "response": response, "kwargs": kwargs, "raw_kwargs": raw_kwargs})
        return response

    def _generate(self, prompt, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, **kwargs)
        if self.drop_prompt_from_output:
            input_length = inputs.input_ids.shape[1]
            outputs = outputs[:, input_length:]
        completions = [{"text": c} for c in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
        return {"prompt": prompt, "choices": completions}

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.request(prompt, **kwargs)
        return [c["text"] for c in response["choices"]]

def initialize_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map='auto',
        quantization_config=bnb_config,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    llama = HFModel(model, tokenizer)
    dspy.settings.configure(lm=llama, max_output_tokens=MAX_OUTPUT_TOKENS)

class QASignature(dspy.Signature):
    query = dspy.InputField(desc="The initial query")
    data = dspy.InputField(desc="The tabular data containing text and numerical columns")
    result = dspy.OutputField(desc="The result based on the query and data in a concise and structured way!")

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(QASignature)

    def forward(self, query, data):
        return self.prog(query=query, data=data)

def chatbot(query, data):
    data_str = dataframe_to_string(data)
    cot = CoT()
    result = cot(query=query, data=data_str)
    return result
