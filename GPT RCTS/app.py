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
Answer Whatever Asked
<</SYS>>
{prompt}[/INST]
'''
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    return tokenizer.decode(output[0])

if __name__ == '__main__':
    # Run the app on port 6030
    app.run(debug=True, port=6030)
