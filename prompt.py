SYSTEM_PROMPT_FORMAT = '''You are actively participating in an experiment called the Bavelas with {} other participants. You need to collaborate with them to finish a task. Here are the task rules: 
    Each participant is given a card upon which has been printed {} symbols taken from a master set of eight symbols. These symbols are a circle, a heart, a triangle, an asterisk, a square, a spade, a plus sign, and a club. 
    Although each symbol appeared on some groups of participants' cards, only one symbol appears on all participants' cards. 
    The group's task is to find the common symbol in the shortest time possible. However, each participant can only communicate with a limited range of participants. Therefore, some information needs to be passed on through intermediaries. You need to pass your information to the other participants and gather information from them. 
    you can infer the answer correctly after you have gathered information from all other participants. Therefore, you need to fully communicate with other participants before submitting your answer. Once you receive replies from one participant, you can ask further questions or reach out to other participants you have access to.
    Crucial information is enclosed in tags including messages, identity, symbols and collaborators. You should stick to your information and never make up any information about your card throughout the whole experiment.
'''
INITIATE_PROMPT = '''Now you can ask the other participants questions or share your information. Your questions and replies should be in JSON format with keys "Recipient" and "Content". "Recipient" is the ID ,which is an integer, of the participant you want to send the message to. "Content" is a string representing the message you want to convey to that recipient. If you want to submit the answer, you need to set the "Recipient" to 0. 
Remember to keep your questions in valid format and never make up information about your card.
Here are some examples of valid questions and replies:
Valid question:
{
    "Recipient": {{an integer X}},
    "Content": "What symbols do you have on your card?"
}, 
Valid reply:
{
    "Recipient": {{an integer Y}},
    "Content": "I have {{fill in with your own information}} on my card."
}
A valid submission would look like this:
{
    "Recipient": 0,
    "Content": "I think the shared symbol is {{fill in with possible answer}}."
}
Now let's start the experiment.  
'''

ASSESS_PROMPT = '''Here is the message from participant {}: <message> {} </message>. Make a classification whether this message contains information about the other participant or not, such as their card information and their relative position in the communication network. If the message is only about asking for your information, then this message is not informative. If you think this message contains the other participant's information, reply "Yes, it contains the other participant's information", otherwise reply "No, it does not contain the other participant's information".'''

SUMMARIZE_PROMPT = '''Here is the message from participant {}: <message> {} </message>. Here are your inferences about other participants:
<inferences> {} </inferences>. Please paraphrase this message into inferences about other participants from your perspective. These inferences include other participants' card information and their relative position in the communication network.
Do not repeat any information that has already been mentioned in your previous inferences. Do not add your future plans, uncertain or unknown things into your inferences. You should also not mention that there is no available information.
Sometimes the message from one persons contains information from another participant. You should figure out which participant this information is about and what the information is.

You should list your inferences in numbers. Each inference you derived should starts with "Participant X" indicating the subject of this inference. Try to condense your reply and cover as much relevant information as possible in one single sentence.'''


REFLECT_PROMPT = '''This round of experiment has ended. The answer you gave was {}. Now you should reflect on the whole round of experiment and extract some useful experience or strategies that could be helpful in the following rounds. 
The communication network your team is using is among circle, chain, star, and complete structure. 
In a circle structure, each node is connected to exactly two other nodes, forming a closed loop or ring. 
In a chain structure, nodes are connected in a linear sequence, with each node linked to at most two other nodes (except for the endpoints).
In a star structure, one central node (hub) is connected to all other nodes, while the other nodes are not directly connected to each other. 
In a complete structure, every node is connected to every other node directly.
You can infer what kind of structure your team is using. Keep you reply simple and in less than 20 words.'''

EVALUATE_PROMPT = '''Here are your inferences about other participants: 
<inference>
{}
</inference>.
Combining them with your own card information: 
<personal information> {} </personal information>, {} think about the following question:
Do you have all the information you need to infer the answer? If not, what extra information do you need, and which participant should you contact to get that information? 
Your reply should be less than 100 words.'''

EVALUATE_PROMPT_WITHOUT_INFERENCE = '''Consider your own card information: 
<personal information> {} </personal information>, your chat history, {} think about the following question:
Do you have all the information you need to infer the answer? If not, what extra information do you need, and which participant should you contact to get that information? 
Your reply should be less than 100 words.'''


PLANNING_PROMPT = ''' Here is the message from participant {}: <message> {} </message> {} 
Your message should be in JSON format with keys "Recipient" and "Content". 
"Recipient" is the ID, which is an integer, of the participant you want to send the message to. 
"Content" is a string representing the content you want to convey to that recipient. If you want to submit the answer, you need to set the "Recipient" to 0. You can only submit one symbol in each round.'''

IDENTITY_FORMAT = '''You are <identity> participant {} </identity>. There are <symbols> {} </symbols> on your card. The participants you can contact are <collaborator> {} </collaborator>.'''

SIMPLE_PLANNING_PROMPT = '''Here is the message from participant {}: <message> {} </message>. Here is your personal information: <personal information> {} </personal information>. If this is an inquiry, you should answer it based on you personel information and chat history. Otherwise, you need to decide the message you want to send to other participant. 
Your message should be in JSON format with keys "Recipient" and "Content". 
"Recipient" is the ID, which is an integer, of the participant you want to send the message to. 
"Content" is a string representing the content you want to convey to that recipient. If you want to submit the answer, you need to set the "Recipient" to 0.'''

COT_PLANNING_PROMPT = '''Here is the message from participant {}: <message> {} </message>. Here is your personal information: <personal information> {} </personal information>. 
If this is an inquiry, you should answer it based on you personel information and chat history. Otherwise, you need to decide the message you want to send to other participant.
Before you make the decision, you should think step by step about the following questions:
What information do you have right now? What other information do you need to infer the final answer? Who do you need to contact to get that information?
Your message should be in JSON format with keys "Recipient" and "Content". 
"Recipient" is the ID, which is an integer, of the participant you want to send the message to. 
"Content" is a string representing the content you want to convey to that recipient. If you want to submit the answer, you need to set the "Recipient" to 0.'''

SIMPLE_CONSISTENT_PLANNING_PROMPT = '''Here is the message from participant {}: <message> {} </message>. Here is your personal information: <personal information> {} </personal information>. If this is an inquiry, you should answer it based on you personel information and chat history. Otherwise, you need to decide the message you want to send to other participant. 
Think of one possible plan to derive the final answer, your reply should be less than 100 words.'''

SELF_CONSISTENCY_PROMPT = '''Here is the message from participant {}: <message> {} </message>. Here is your personal information: <personal information> {} </personal information>.
Here are some possible plans of responsing to this message: 
<possible response> {} </possible response>.
Extract the common elements from these possible responses and use these elements to derive the final 
Your message should be in JSON format with keys "Recipient" and "Content". 
"Recipient" is the ID, which is an integer, of the participant you want to send the message to. 
"Content" is a string representing the content you want to convey to that recipient. If you want to submit the answer, you need to set the "Recipient" to 0.'''


REJECT_PROMPT = '''Sorry, you cannot contact participant {}. You can only contact {}. Please consider this constraint and send another message.'''

ASK_PROMPT = '''Some of your teammates have submitted the answer. You can either ask other participants a question or submit a possible answer. Your message should be in JSON format with keys "Recipient" and "Content". 
"Recipient" is the ID, which is an integer, of the participant you want to send the message to. 
"Content" is a string representing the content you want to convey to that recipient. If you want to submit the answer, you need to set the "Recipient" to 0. You can only submit one symbol in each round.'''

INPUT_TEMPLATE = '''You are <identity> participant {} </identity>. There are <symbols> {} </symbols> on your card. The participants you can contact are <collaborator> {} </collaborator>. Now you can ask other participants question for their information or talk to others about your information. Remember to keep your questions in valid format and never make up information about your card.'''

