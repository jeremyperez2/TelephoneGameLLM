# This file is used to get the answer from the server. 
#It is called by the agent.py file.
import requests
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_answer(access_url, prompt, debug=False, instruct = True, start_flag = None, use_vllm = True, model = None, sampling_params=None, temperature = 0.8):
    if use_vllm:
        from vllm import SamplingParams

        if instruct:
                
            tokenizer = model.get_tokenizer()

            conversations = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                tokenize=False,
            )
        else:
            conversations = prompt + 'Assistant: Sure, here is the requested answer:\n\n1.'

        output = model.generate(
            [conversations],
            SamplingParams(
                temperature=temperature,
                top_p=0.95,
                max_tokens=None,
                stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            )
        )
        #output = model.generate([prompt], sampling_params=sampling_params)
        return output[0].outputs[0].text

    if instruct:
        url = access_url + "/v1/chat/completions"
    else:
        url = access_url + "/v1/completions"

    headers = {
        "Content-Type": "application/json"
    }

    history = []

    #prompt = '<|im_start|>user' + prompt + '<|im_end|> <|im_start|>assistant'
    if instruct:
        prompt = prompt + 'Here is the requested answer:\n\n1.'
        history.append({"role": "user", "content": prompt})
        data = {
            "mode": "instruct",
            "role": "Empty",
            "messages": history
        }
    else:
        prompt = prompt + 'Assistant: Sure, here is the requested answer:\n\n1.'
        if start_flag != None: 
            prompt = prompt + start_flag

        data = {
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": temperature



        }

    while True:
        response = requests.post(url, headers=headers, json=data, verify=False)
        if instruct:
            try:
                assistant_message = response.json()['choices'][0]['message']['content'].replace('</s>', '')
                #print("Answer: " +assistant_message)
                break
            except:
                print('No answer from server, trying again...')
        else:
            assistant_message = response.json()['choices'][0]['text']
            #print(assistant_message)
            

    #history.append({"role": "assistant", "content": assistant_message})
    #stories.append("Story" +str(i)+": " + assistant_message)
    #print("Answer: " +assistant_message)

    return assistant_message



  
# if __name__ == "__main__":
#     print(get_answer('https://sides-create-born-institute.trycloudflare.com', "Tell me a joke"))