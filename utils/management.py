from openai import OpenAI
import anthropic
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


# def claude_completion(client, model, msgs, temperature, max_tokens)

def openai_completion(client, model, msgs, temperature=0.4, reply_number=1, json_mode=True):
    return client.chat.completions.create(
                        model=model,
                        messages=msgs,
                        temperature = temperature,
                        n = reply_number,
                        response_format={"type": "json_object"} if json_mode else None
                    )

def parse_openai(completion, self_consistency=False):
    if self_consistency:
        contents = []
        for choice in completion.choices:
            contents.append(choice.message.content)
    else:
        contents = completion.choices[0].message.content
    usage = completion.usage
    return contents, usage

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

        if self.model == 'claude-3-5-sonnet-20240620':
            self.client = anthropic.Anthropic(api_key=API_KEY)
        else:
            self.client = OpenAI(api_key=API_KEY)
        # print(SYSTEM_PROMPT)
        self.output_path = output_path
        self.logging_path = output_path + '/log.txt'
        logging.basicConfig(filename=self.logging_path, level=logging.INFO)
        self.agents = [
            [] for _ in range(self.num_agents)
        ]
        # self.tokens = [ 
        #     [] for _ in range(self.num_agents)
        # ]

        self.beliefs = [
            [
                [] for _ in range(self.num_agents)
            ] for _ in range(self.num_agents)
        ]

        self.reflection = [
            [] for _ in range(self.num_agents)
        ]
        self.identity = []
        self.agent_contact = []


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
    
    def check_valid(self, msg):
        source, target, content = msg.source, msg.target + 1, msg.content
        if target == 0:
            return True
        if 'all' in self.agent_contact[source]:
            return True
        if str(target) in self.agent_contact[source]:
            return True
        return False

    def reject(self, msg, last_prompt = None, REJECT_PROMPT=None):
        if REJECT_PROMPT == None:
            REJECT_PROMPT = '''Sorry, you cannot contact participant {}. You can only contact {}. Please consider this constraint and send another message.'''
        source, target, content = msg.source, msg.target, msg.content
        print("The message from {} to {}: {} is rejected.".format(source + 1, target + 1, content))
        reject_input = REJECT_PROMPT.format(target + 1, self.agent_contact[source])
        print(reject_input)
        if last_prompt is not None:
            reject_input += last_prompt
        msg_formatted = [{"role": "user", "content": reject_input}]
        response, usage = parse_openai(openai_completion(self.client, self.model, self.agents[source]+ msg_formatted))
        print("new message: {}".format(response))
        return response

    def generate_agents(self, agent_config: list, INITIATE_PROMPT, IDENTITY_FORMAT, num_signs: int):
        assert len(agent_config) == self.num_agents
        
        for idx in range(self.num_agents):
            no, card, contact = agent_config[idx]['no'], agent_config[idx]['card'], agent_config[idx]['links']
            self.agent_contact.append(agent_config[idx]['links'])
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
        response = response.strip()
        try:
            response = eval(response)
        except:
            print("response: ", response)
            exit()
        assert type(response) == dict, TypeError
        recipient = response["Recipient"]
        content = response["Content"]
        assert type(recipient) == int, TypeError
        return recipient, content
    
    def get_inference(self, idx, latest_k=5):
        belief_history = self.beliefs[idx]
        BELIEF_FORMAT = '''For Participant {}, the inferences are: {} \n'''
        total_belief = ''
        for i in range(self.num_agents):
            if i == idx:
                continue
            if len(belief_history[i]) == 0:
                belief = '''For Participant {}, you have no inference right now, you should collect some information about him. \n'''.format(i+1)
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
                # self.tokens.append(usage)
                recipient, content = self.parse_response(content)
                is_valid = self.check_valid(Message(source=i, target=recipient-1, content=content))
                while not is_valid:
                    response = self.reject(Message(source=i, target=recipient-1, content=content))
                    recipient, content = self.parse_response(response)
                    is_valid = self.check_valid(Message(source=i, target=recipient-1, content=content))
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
        # self.tokens.append(usage)
        summaries = ""
        is_inquiry = True
        if "yes" in reply.lower():
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
                self.reduce_redundency(target)
        return is_inquiry, summaries
    
    def reduce_redundency(self, idx):
        s = set()
        for x in self.beliefs[idx]:
            for y in x:
                if y not in s:
                    s.add(y)
                else:
                    x.remove(y)
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
        total_belief = self.get_inference(idx)
        if len(self.reflection[idx]) == 0:
            input = EVAL_PROMPT.format(total_belief, self.identity[idx], "")
        else:
            REFLECTION_FORMAT = '''and your reflections from former rounds: <reflection> {} </reflection>'''.format(' '.join(self.reflection[idx]))
            input = EVAL_PROMPT.format(total_belief, self.identity[idx], REFLECTION_FORMAT)
        logger.info("Evaluation input: {}".format(input))
        # print("Evaluation input: {}".format(input))
        content, usage = parse_openai(openai_completion(self.client, self.model, self.agents[idx] + [{"role": "user", "content": input}], json_mode=False))
        logger.info("Evaluation reply: {}".format(content))
        # print("Evaluation reply: {}".format(content))
        return content
    def planning(self, PLANNING_PROMPT, msg, evaluation, is_evaluation, is_inquiry, with_inference=True):

        total_belief = self.get_inference(msg.target)

        source, target, content = msg.source, msg.target, msg.content
        if is_inquiry:
            prompt = '''Please answer this inquiry combining with your own information 
            <personal information> {} </personal information> 
            and your inferences about other participants
            <inferences> {} </inferences>.'''.format(self.identity[msg.target], total_belief)
        elif is_evaluation:
            prompt = '''Please consider this message, your own information 
            <personal information> {} </personal information>
            and the evaluation you've made before: 
            <evaluation> {} </evaluation> 
            to determine the message you want to send to other participants.'''.format(self.identity[msg.target], evaluation)
        else:
            prompt = '''Please consider this message, your own information 
            <personal information> {} </personal information>
            and your inferences about other participants
            <inferences> {} </inferences>
            to determine the message you want to send to other participants.'''.format(self.identity[msg.target], total_belief)
        if with_inference:
            input = PLANNING_PROMPT.format(source+1, content, prompt)
        else:
            input = PLANNING_PROMPT.format(source+1, content, self.identity[msg.target])
        logger.info("Planning input: {}".format(input))
        # print ("Planning input: {}".format(input))
        response, usage = parse_openai(openai_completion(self.client, self.model, self.agents[target] + [{"role": "user", "content": input}], json_mode=True))
        logger.info("Planning reply: {}".format(response))
        # print("Planning reply: {}".format(response))
        recipient, content = self.parse_response(response)
        is_valid = self.check_valid(Message(source=target, target=recipient-1, content=content))
        while not is_valid:
            new_msg = self.reject(Message(source=target, target=recipient-1, content=content), last_prompt=input)
            recipient, content = self.parse_response(new_msg)
            is_valid = self.check_valid(Message(source=target, target=recipient-1, content=content))
        return Message(source=target, target=recipient-1, content=content), Message(source=target, target=recipient-1, content=response)
    
    
    def self_consistency_planning(self, SIMPLE_PLANNING_PROMPT, SELF_CONSISTENCY_PROMPT, msg):
        response_buffer = []
        source, target, content = msg.source, msg.target, msg.content
        input = SIMPLE_PLANNING_PROMPT.format(source+1, content, self.identity[msg.target])

        responses, usage = parse_openai(openai_completion(self.client, self.model, self.agents[msg.target] + [{"role": "user", "content": input}], temperature=1.0, reply_number=3, json_mode=None), self_consistency=True)
        # print(responses)
        options = ""
        for i in range(len(responses)):
            options += '{}. {}\n'.format(i+1, responses[i])
        # print (options)
        selection = SELF_CONSISTENCY_PROMPT.format(source+1, content, self.identity[msg.target], options)

        response, usage = parse_openai(openai_completion(self.client, self.model, self.agents[target] + [{"role": "user", "content": selection}], json_mode=True))
        recipient, content = self.parse_response(response)
        is_valid = self.check_valid(Message(source=target, target=recipient-1, content=content))
        while not is_valid:
            new_msg = self.reject(Message(source=target, target=recipient-1, content=content), last_prompt=selection)
            recipient, content = self.parse_response(new_msg)
            is_valid = self.check_valid(Message(source=target, target=recipient-1, content=content))
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
        output = []
        for i in range(self.num_agents):
            self.agents[i].append({"role": "user", "content": REFLECT_PROMPT})
            content, _ = parse_openai(openai_completion(self.client, self.model, self.agents[i], json_mode=False))
            self.reflection[i].append(content)
            print("reflection: {}".format(content))
            output.append(content)
        return output

    def ask(self, ASK_PROMPT, idx):
        input_formatted = {"role": "user", "content": ASK_PROMPT}
        self.agents[idx].append(input_formatted)
        content, _ = parse_openai(openai_completion(self.client, self.model, self.agents[idx], json_mode=False))
        self.agents[idx].append({"role": "assistant", "content": content})
        recipient, content = self.parse_response(content)
        print("Manager asks participant {} to submit the answer".format(idx+1))
        return Message(source=idx, target=recipient-1, content=content)

    def save(self, path, save_beliefs=True):
        print(f"[utils.py] [AgentDialogManagement] saving {path}.pkl ...")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(self.agents, f)
        print(f"[utils.py] [AgentDialogManagement] saving {path}_token.pkl ...")
        # with open(f"{path}_token.pkl", "wb") as f:
        #     pickle.dump(self.tokens, f)
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
        # self.tokens = [ 
        #     [] for _ in range(self.num_agents)
        # ]
        self.beliefs = [
            [
                [] for _ in range(self.num_agents)
            ] for _ in range(self.num_agents)
        ]
        self.identity = []
        # self.reflection = [
        #     [] for _ in range(self.num_agents)
        # ]
        self.agent_contact = []

if __name__ == '__main__':
    print("ok")


# def send_message(self, idx, model:str=None, temperature:float=0.2, max_new_tokens:int=1000):
#     idx:list = self._check_idx(idx)
#     if model is None:
#         model = self.model
#     cur_cnt = 0
#     memory = []
#     summary_log = None
#     RETRY_CNT = 0
#     while cur_cnt < len(idx):
#         if RETRY_CNT >= 20:
#             self._print_log("exceed the max count of retrying")
#             return None
#         try:
#             index = idx[cur_cnt]       
#             if summary_log is None:
#                 summary_log = {'agent': cur_cnt, 'summary?':False, 'idx': 3, 'copy': None}
#             assert self.agents[index][-1]["role"] == "user"
#             # self._print_log(self.agents[index])
#             if not SIMULATE_OPENAI:
#                 completion = self.client.chat.completions.create(
#                     model=model,
#                     messages=self.agents[index],
#                     temperature = temperature,
#                     n=1,
#                     response_format={ "type": "json_object" }
#                 )
#             else:
#                 completion = simulate_openai()
#             memory.append(completion)
#             cur_cnt += 1
#             if summary_log['summary?']:
#                 self.agents[index] = summary_log['copy']
#             summary_log = None
#         except Exception as e:
#             self._print_log(e)
#             if "maximum context length is 4097 tokens" in str(e) or "Your input is too long.".lower() in str(e).lower():
#                 return None
#             self._print_log(f"waiting for {self.RETRY_TIME} second...")
#             time.sleep(self.RETRY_TIME)
#     return memory

    # def answer_phase(self, model: str=None):
    #     if model is None:
    #         model = self.model
    #     reply_buffer = []
    #     for msg in self.buffer:
    #         source, recipient, content = msg.source, msg.target, msg.content
    #         msg_formatted = [{"role": "user", "content": "Message from participant {}: {}".format(source + 1, content)}]
    #         completion = self.client.chat.completions.create(
    #                         model=model,
    #                         messages=self.agents[recipient] + msg_formatted,
    #                         n=1,
    #                         temperature = 0.2,
    #                         response_format={ "type": "json_object" }
    #                     )
    #         content = completion.choices[0].message.content
    #         print(content)
    #         sendto, content = self.parse_response(content)
    #         reply_buffer.append(Message(source=recipient, target=source, content=content))

    #     # add the question and the answer to the history of the respondent
    #     # for q, a in zip(self.buffer, reply_buffer):
    #     #     # assert q.source == a.target and q.target == a.source , "question and answer unmatched"
    #     #     self.agents[q.target].append({"role": "user", "content": "Message from participant {}: {}".format(q.source + 1, q.content)})
    #     #     self.agents[q.target].append({"role": "assistant", "content": "{}".format(a.content)})
    #     # self.buffer = []
    #     #TODO still got a problem: the initial question got too far away from its answer.
    #     for msg in reply_buffer:
    #         self.agents[msg.target].append({"role": "user", "content": "Message from participant {}: {}".format(msg.source + 1, msg.content)})
    #     return reply_buffer

    # def parse_message(self, idx, memory: list): 
    #     idx:list = self._check_idx(idx)
    #     assert len(idx) == len(memory)
    #     msgs: list[Message] = []
    #     for cnt, index in enumerate(idx):
    #         assert self.agents[index][-1]["role"] == "user"
    #         content = memory[cnt].choices[0].message.content
    #         # print(content, type(content))
    #         self.agents[index].append(
    #             {"role": "assistant", "content": content}
    #         )
    #         self.tokens[index].append(
    #             memory[cnt].usage
    #         )
    #         recipient, content = self.parse_response(content)
    #         self.buffer.append(Message(source=index, target=recipient-1, content=content))
    #         msgs.append(Message(source=index, target=recipient-1, content=content))
    #     return msgs