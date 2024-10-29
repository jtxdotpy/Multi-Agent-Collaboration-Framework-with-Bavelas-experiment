from openai import OpenAI
import numpy as np
import time
import logging
logger = logging.getLogger(__name__)
from typing import NamedTuple
import pickle
import re

SIMULATE_OPENAI = False

OPENAI_API = 'Replace_With_Your_Own_API_Key'
MODEL = "gpt-4o"

def openai_completion(client, model, msgs, temperature=0.4, reply_number=1, json_mode=True):
    return client.chat.completions.create(
                        model=model,
                        messages=msgs,
                        temperature = temperature,
                        n = reply_number,
                        response_format={"type": "json_object"} if json_mode else None
                    )

def parse_openai(completion):
    content = completion.choices[0].message.content
    usage = completion.usage
    return content, usage

def parse_summary(summary):
    items = summary.split('\n')
    items = [item.strip() for item in items if item.strip()]
    items = [' '.join(item.split(' ')[1:]) for item in items]
    return items


def simulate_openai():
    return {
     'id': 'chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve',
     'object': 'chat.completion',
     'created': 1677649420,
     'model': 'gpt-3.5-turbo',
     'usage': {'prompt_tokens': 56, 'completion_tokens': 31, 'total_tokens': 87},
     'choices': [
       {
        'message': {
          'role': 'assistant',
          'content': f'This is a test. {time.ctime()}'},
        'finish_reason': 'stop',
        'index': 0
       }
      ]
    }

class Message(NamedTuple):
    source: int     ### start from 0 to num_agents - 1
    target: int     ### start from 0 to num_agents - 1
    content: str
    # reply: str = None

class AgentDialogManagement:
    def __init__(
        self,
        num_agents: int,
        model: str,
        API_KEY:str,
        ORGNIZATION: str=None,
        RETRY_TIME: int=180,
        SYSTEM_PROMPT: str=None,
        output_path: str=None,
        # SYSTEM_PROMPT: str=f"You are ChatGPT, a large language model trained by OpenAI. Knowledge cutoff: 2021-09 Current date: {time.strftime('%Y-%m-%d')}"
    ):
        
        self.num_agents = num_agents
        self.model = model
        self.RETRY_TIME = RETRY_TIME - np.random.randint(int(RETRY_TIME*0.8))
        print(f"Wait for {self.RETRY_TIME} seconds if timeout occurs")
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.client = OpenAI(api_key=OPENAI_API)
        # print(SYSTEM_PROMPT)
        self.output_path = output_path
        self.logging_path = output_path + '/log.txt'
        logging.basicConfig(filename=self.logging_path, level=logging.INFO)
        self.agents = [
            [] for _ in range(self.num_agents)
        ]
        self.tokens = [ 
            [] for _ in range(self.num_agents)
        ]
        self.buffer: list[Message] = []
        self.beliefs = [
            [
                [] for _ in range(self.num_agents)
            ] for _ in range(self.num_agents)
        ]

        self.reflection = [
            [] for _ in range(self.num_agents)
        ]
        self.identity = []

        # if default_model.lower() in MODEL_MAPPING:
        #     self.model = MODEL_MAPPING[default_model](api_key=API_KEY)
        #     print(f'Backbone: {default_model}')
        #     # "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        #     # 'meta/llama-2-13b-chat:56acad22679f6b95d6e45c78309a2b50a670d5ed29a37dd73d182e89772c02f1',
        #     if default_model.lower().startswith('replicate') and 'llama70' in default_model.lower():
        #         print(f'Backbone: ReplicateLlaMA70')
        #         ReplicateLlaMA.model_id = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
        #     elif default_model.lower().startswith('replicate') and 'llama13' in default_model.lower():
        #         print(f'Backbone: ReplicateLlaMA13')
        #         ReplicateLlaMA.model_id = 'meta/llama-2-13b-chat:56acad22679f6b95d6e45c78309a2b50a670d5ed29a37dd73d182e89772c02f1'
        # else:
        print(f"Backbone: OpenAI-{model}")

        self.terminate = False

    def _print_log(self, message):
        print(f"[{time.ctime()}] {message}")

    def _check_idx(self, idx):
        if isinstance(idx, list):
            assert len(idx) >= 1
        if isinstance(idx, int):
            assert idx >= 0 and idx < self.num_agents
            idx = [idx]
        if isinstance(idx, str):
            assert idx.lower() == "all"
            idx = list(range(self.num_agents))
        return idx

    def generate_agents(self, agent_config: list, INITIATE_PROMPT, IDENTITY_FORMAT, num_signs: int):
        assert len(agent_config) == self.num_agents
        
        for idx in range(self.num_agents):
            no, card, contact = agent_config[idx]['no'], agent_config[idx]['card'], agent_config[idx]['links']
            personal_info = IDENTITY_FORMAT.format(no, card, contact)
            self.identity.append(personal_info)
            if self.SYSTEM_PROMPT is not None:
                self.agents[idx].append(
                    {
                        "role": "system",
                        "content": self.SYSTEM_PROMPT.format(self.num_agents-1, num_signs)
                    }
                )
            self.agents[idx].append(
                {
                    "role":"user",
                    "content": personal_info + INITIATE_PROMPT
                }
            )
        print("Agent generated")

    def parse_response(self, response):
        response = eval(response)
        assert type(response) == dict, TypeError
        recipient = response["Recipient"]
        content = response["Content"]
        assert type(recipient) == int, TypeError
        return recipient, content


    def parse_message(self, idx, memory: list): 
        idx:list = self._check_idx(idx)
        assert len(idx) == len(memory)
        msgs: list[Message] = []
        for cnt, index in enumerate(idx):
            assert self.agents[index][-1]["role"] == "user"
            content = memory[cnt].choices[0].message.content
            # print(content, type(content))
            self.agents[index].append(
                {"role": "assistant", "content": content}
            )
            self.tokens[index].append(
                memory[cnt].usage
            )
            recipient, content = self.parse_response(content)
            self.buffer.append(Message(source=index, target=recipient-1, content=content))
            msgs.append(Message(source=index, target=recipient-1, content=content))
        return msgs
    
    def get_inference(self, idx):
        belief_history = self.beliefs[idx]
        BELIEF_FORMAT = '''For Paticipant {}, the inferences are: {} \n'''
        total_belief = ''
        for i in range(self.num_agents):
            if i == idx:
                continue
            if len(belief_history[i]) == 0:
                belief = '''For Paticipant {}, you have no inference right now, you should collect some information about him. '''.format(i+1)
                total_belief += belief
            else:
                belief = BELIEF_FORMAT.format(i+1, '\n'.join(belief_history[i]))
                total_belief += belief
        return total_belief
    
    def initiate_chat(self):
        msgs = []
        i = 0
        while i < self.num_agents:
            try:
                content, usage = parse_openai(openai_completion(self.client, self.model, msgs=self.agents[i]))
                self.agents[i].append(
                    {"role": "assistant", "content": content}
                )
                self.tokens.append(usage)
                recipient, content = self.parse_response(content)
                self.buffer.append(Message(source=i, target=recipient-1, content=content))
                msgs.append(Message(source=i, target=recipient-1, content=content))
                i += 1
            except Exception as e:
                self._print_log(e)
                if "maximum context length is 4097 tokens" in str(e) or "Your input is too long.".lower() in str(e).lower():
                    return None
                self._print_log(f"waiting for {self.RETRY_TIME} second...")
                time.sleep(self.RETRY_TIME)
        return msgs
    
    def assess(self, msg, ASSESS_PROMPT, SUMMARIZE_PROMPT):
        source, target, content = msg.source, msg.target, msg.content
        assess_input = ASSESS_PROMPT.format(source+1, content)
        logger.info("Assessment input: {}".format(assess_input))
        
        msg_formatted = {"role": "user", "content": assess_input}
        ### don't add this conversation into history
        # self.agents[target].append(msg_formatted)
        completion = openai_completion(self.client, self.model,self.agents[target] + [msg_formatted], json_mode=False)
        reply, usage = parse_openai(completion)
        logger.info("Assessment reply: {}".format(reply))
        
        # self.agents[target].append(
        #     {"role": "assistant", "content": content}
        # )
        self.tokens.append(usage)
        summaries = ""
        is_inquiry = True
        if "statement" in reply.lower():
            is_inquiry = False
            summaries = self.summarize(msg, target, SUMMARIZE_PROMPT)
            for summary in summaries:
                try:
                    subject = eval(summary.split()[1])
                    if (type(subject) != int) or not (subject >=1 and subject <= self.num_agents):
                        subject = source
                except:
                    subject = source
                self.beliefs[target][subject-1].append(summary)
        return is_inquiry, summaries

    def summarize(self, msgs, agent_idx, SUMMARIZE_PROMPT):
        source, target, content = msgs.source, msgs.target, msgs.content
        total_belief = self.get_inference(agent_idx)
        input = SUMMARIZE_PROMPT.format(source + 1, content, total_belief)
        logger.info("Summarize input: {}".format(input))
        # print("Summarize input: {}".format(input))
        msg_formatted = {"role": "user", "content": input}
        summary, _ = parse_openai(openai_completion(self.client, self.model, self.agents[agent_idx] + [msg_formatted], json_mode=False))
        summaries = parse_summary(summary)
        logger.info("Summarize reply: {}".format(summary))
        # print("Summarize reply: {}".format(summary))
        return summaries

    def evaluate(self, EVAL_PROMPT, idx):
        # belief_history = self.beliefs[idx]
        # BELIEF_FORMAT = '''For Paticipant {}, the inferences are: {} \n'''
        # total_belief = ''
        # for i in range(self.num_agents):
        #     if i == idx:
        #         continue
        #     if len(belief_history[i]) == 0:
        #         belief = '''For Paticipant {}, you have no inference right now, you should collect some information about him. '''.format(i+1)
        #         total_belief += belief
        #     else:
        #         belief = BELIEF_FORMAT.format(i+1, '\n'.join(belief_history[i]))
        #         total_belief += belief
        total_belief = self.get_inference(idx)
        if len(self.reflection[idx]) == 0:
            input = EVAL_PROMPT.format(total_belief, self.identity[idx], "")
        else:
            REFLECTION_FORMAT = '''and your reflections from former rounds: {}'''.format(' '.join(self.reflection[idx]))
            input = EVAL_PROMPT.format(total_belief, self.identity[idx], REFLECTION_FORMAT)
        logger.info("Evaluation input: {}".format(input))
        # print("Evaluation input: {}".format(input))
        content, usage = parse_openai(openai_completion(self.client, self.model, self.agents[idx] + [{"role": "user", "content": input}], json_mode=False))
        logger.info("Evaluation reply: {}".format(content))
        # print("Evaluation reply: {}".format(content))
        return content
    def planning(self, PLANNING_PROMPT, msg, evaluation, is_inquiry):

        total_belief = self.get_inference(msg.target)

        source, target, content = msg.source, msg.target, msg.content
        if is_inquiry:
            prompt = '''Please answer this inquiry combining with your own information 
            <personal information> {} </personal information> 
            and your inferences about other participants
            <inferences> {} </inferences>.
            Please do not ask further question.'''.format(self.identity[msg.target], total_belief)
        else:
            prompt = '''Please consider this message, your own information 
            <personal information> {} </personal information>
            and the evalutation you've made before: 
            <evaluation> {} </evaluation> 
            to determine the message you want to send to other participants.'''.format(self.identity[msg.target], evaluation)

        input = PLANNING_PROMPT.format(source+1, content, prompt)
        logger.info("Planning input: {}".format(input))
        # print ("Planning input: {}".format(input))
        response, usage = parse_openai(openai_completion(self.client, self.model, self.agents[target] + [{"role": "user", "content": input}], json_mode=True))
        logger.info("Planning reply: {}".format(response))
        # print("Planning reply: {}".format(response))
        recipient, content = self.parse_response(response)
        return Message(source=target, target=recipient-1, content=content), Message(source=target, target=recipient-1, content=response)
    
    def add_to_history(self, idx, msg: Message, response: Message):
        msg_formatted = {"role": "user", "content": "Message from participant {}: {}".format(msg.source + 1, msg.content)}
        self.agents[idx].append(msg_formatted)
        self.agents[idx].append({"role": "assistant", "content": response.content})

    def reflect(self, REFLECT_PROMPT, correct, right_ans):
        outcome = 'right' if correct else 'wrong'
        if not correct:
            outcome += '. The correct answer is {}'.format(right_ans)
        REFLECT_PROMPT = REFLECT_PROMPT.format(outcome)
        for i in range(self.num_agents):
            self.agents[i].append({"role": "user", "content": REFLECT_PROMPT})
            content, _ = parse_openai(openai_completion(self.client, self.model, self.agents[i]))
            self.reflection[i].append(content)

    def save(self, path, save_beliefs=True):
        print(f"[utils.py] [AgentDialogManagement] saving {path}.pkl ...")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(self.agents, f)
        print(f"[utils.py] [AgentDialogManagement] saving {path}_token.pkl ...")
        with open(f"{path}_token.pkl", "wb") as f:
            pickle.dump(self.tokens, f)
        if save_beliefs:
            with open(f"{path}_beliefs.pkl", "wb") as f:
                pickle.dump(self.beliefs, f)

    def print_history(self, index=None):
        if index == None:
            for x in self.agents:
                print(x)
                print('-------------------------')
        else:
            print(self.agents[index])

    def clear_history(self):
        self.agents = [
            [] for _ in range(self.num_agents)
        ]
        self.tokens = [ 
            [] for _ in range(self.num_agents)
        ]
        self.buffer: list[Message] = []
        self.belief = [
            [
                [] for _ in range(self.num_agents)
            ] for _ in range(self.num_agents)
        ]

if __name__ == '__main__':
    print("ok")
