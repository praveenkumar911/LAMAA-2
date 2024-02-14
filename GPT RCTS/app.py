from flask import Flask, render_template, request

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)

model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        response = generate_response(prompt)
        return render_template('index.html', prompt=prompt, response=response)
    else:
        return render_template('index.html', prompt='', response='')

def generate_response(prompt):
    prompt_template = f'''[INST] <<SYS>>
Your are a chatbot trained with the Telangana Meeseva Services. You should only respond accurately, with only what is asked. Don't add any additional information, unless asked for. You should provide answers to the question only related to Telangana Meeseva services and do not use or include other state names like Karnataka in your response. You do not answer any question from outside the Meeseva context. Don't use your prior knowledge.
If the question is from outside the provided document information, then strictly don't answer. If uncertain, indicate that you don't know rather than guessing, and do not try to make up the answer.
<</SYS>>
{prompt}[/INST]
'''
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    return tokenizer.decode(output[0])

if __name__ == '__main__':
    app.run(debug=True)
