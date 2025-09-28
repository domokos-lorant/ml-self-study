import json
import requests
import msal
from rich.console import Console
from rich.markdown import Markdown

class LLMClient:
    _ENDPOINT = 'https://fe-26.qas.bing.net/sdf/'
    _SCOPES = ['https://substrate.office.com/llmapi/LLMAPI.dev']
    _API_COMPLETIONS = 'completions'
    _API_CHAT_COMPLETIONS = 'chat/completions'
    _PRINT_TOOLS = False

    def __init__(self, endpoint):
        if endpoint != None:
            LLMClient._ENDPOINT = endpoint        

        self._app = msal.PublicClientApplication('68df66a4-cad9-4bfd-872b-c6ddde00d6b2',
                                            authority='https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47',
                                            enable_broker_on_windows=True, enable_broker_on_mac=True)

    def send_request(self, model_name, request, chat_completion = False, api_version = None):
        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            'Content-Type':'application/json', 
            'Authorization': 'Bearer ' + token, 
            'X-ModelType': model_name }

        body = str.encode(json.dumps(request))
        _endpoint = LLMClient._ENDPOINT + LLMClient._API_CHAT_COMPLETIONS if chat_completion else LLMClient._ENDPOINT + LLMClient._API_COMPLETIONS
        _endpoint = _endpoint + "?api-version=" + api_version if api_version else _endpoint
        with requests.post(_endpoint, data=body, headers=headers) as response:
            #response.raise_for_status()
            #print(response.json())
            return response.json()

    def _get_token(self):
        accounts = self._app.get_accounts()
        result = None

        if accounts:
            # Assuming the end user chose this one
            chosen = accounts[0]

            # Now let's try to find a token in cache for this account
            result = self._app.acquire_token_silent(LLMClient._SCOPES, account=chosen)
    
        if not result:
            result = self._app.acquire_token_interactive(scopes=LLMClient._SCOPES, parent_window_handle=self._app.CONSOLE_WINDOW_HANDLE)
            
            if 'error' in result:
                raise ValueError(
                    f"Failed to acquire token. Error: {json.dumps(result, indent=4)}"
                )

        return result["access_token"]

def simple_request(llm_client, model_name, prompt):
    request_data = {
            "prompt": prompt,
            "max_tokens":50,
            "temperature":0.6,
            "top_p":1,
            "n":5,
            "stream":False,
            "logprobs":None,
            "stop":"\n"
    }

    response = llm_client.send_request(model_name, request_data)
    print(response)

def chat_request(llm_client, model_name, prompt):
    request_data = {
            "messages":[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens":500,
            "temperature":0.6,
            "top_p":1,
            "n":1,
            "logprobs":None,
            "stop":"\r\n"
    }    

    response = llm_client.send_request(model_name, request_data, chat_completion = True)
    return response

def tool_request(llm_client, model_name, messages):
    for _ in range(5):  # simple tool call limit
        stream_request_data = {
                "messages": messages,
                "max_tokens":1000,
                "temperature":0.1,
                "top_p":1,
                "n":1,
                "stream":False,
                "logprobs":None,
                "tools":TOOLS,
                "tool_choice":"auto",
                "stop":"\r\n"
        }

        response = llm_client.send_request(model_name, stream_request_data, chat_completion = True)
        
        #print(response)

        if "tool_calls" in response["choices"][0]["message"]:
            #print("\n\nTool calls:")
            for tool_call in response["choices"][0]["message"]["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                print(f"\nCalling tool: {tool_name}")
                tool_args = tool_call["function"]["arguments"]
                tool_result = call_tool(tool_name, tool_args)

                # Append the tool result so the model can use it
                messages.append({
                    "role": "assistant",
                    "tool_calls": [  # echo back the tool call metadata (optional but keeps transcript complete)
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_args
                            }
                        }
                    ],
                    "content": None
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": json.dumps(tool_result)
                })
            # continue loop: send tool outputs back to the model for final answer
            continue
        return response   

#####################################
# Tools Section
#####################################

def send_email(recipient, subject, body):
    # Pretend to send an email
    print(f"Sending email to {recipient} with subject '{subject}' and body '{body}'")
    return {"status": "Email sent successfully"}

LOCAL_FUNCTIONS = {
    "send_email": send_email
}

def call_tool(tool_name: str, tool_args_json: str):
    fn = LOCAL_FUNCTIONS.get(tool_name)
    if not fn:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        args = json.loads(tool_args_json or "{}")
        result = fn(**args)
        return result
    except Exception as e:
        return {"error": str(e)}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "send an email.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "the recipient email address",
                },
                "subject": {
                    "type": "string",
                    "description": "the subject of the email"
                },
                "body": {
                    "type": "string",
                    "description": "the body of the email"
                }
            },
            "required": ["recipient", "subject", "body"]
        }
    }
}]

#####################################
# Main / Chat Loop
######################################
if __name__ == "__main__":
    endpoint = 'https://fe-26.qas.bing.net/sdf/'

    llm_client = LLMClient(endpoint)

    messages = [{
        "role": "system",
        "content": "You are a helpful assistant.",
        "role": "assistant",
        "content": "How can I help you today?",
        }]

    # To render MD
    console = Console()

    console.print("How can I help you today? (Enter to quit)")
    
    # Chat Loop
    while user_request := input("> "):
        console.print()

        # Append new message to existing
        messages.append(
                    {                    
                        "role": "user", 
                        "content":user_request
                    })   

        with console.status("Thinking..."):
            response = tool_request(llm_client, model_name = 'dev-gpt-4o-gg', messages=messages)
        
        console.print(Markdown(response['choices'][0]['message']['content']))
