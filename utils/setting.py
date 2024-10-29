import random
import itertools

def create_cards(num_agents, num_signs=3, signs=None):
    ### make sure there is one and only one sign appearing on every card.
    if signs == None:
        signs = ['a triangle', 'an asterisk', 'a square', 'a plus sign', 'a club', 'a circle', 'a heart', 'a spade']
    cards = []
    allow_repetition = num_agents - 1
    num_left = {k: allow_repetition for k in signs}
    
    same_sign = random.choice(signs)
    signs.remove(same_sign)
    cbn = list(itertools.combinations(signs, num_signs - 1))
    random.shuffle(cbn)
    j = 0
    # print(sample_cbn)
    for i in range(num_agents):
        while True:
            other = list(cbn[j])
            qualify = True
            for item in other:
                if num_left[item] == 0:
                    qualify = False
                    break
            if not qualify:
                j += 1
            else:
                j += 1
                for item in other:
                    num_left[item] -= 1
                break
        
        card = [same_sign]
        card.extend(other)
        
        random.shuffle(card)
        sent = ', '.join(card[0:-1]) + ' and ' + card[-1]
        cards.append(sent)
    
    answer = same_sign.split(' ')[1] if same_sign != 'a plus sign' else 'plus sign'
    return cards, answer

def create_topology(num_agents, topology='circle', center=None):
    if num_agents == 2:
        return ['participant 2', 'participant 1']
    contact = []
    agents = list(range(1, num_agents+1))
    if topology == 'circle':
        for i in range(num_agents):
            contact.append('participant {} and participant {}'.format(agents[i-1], agents[(i+1)%num_agents]))
    elif topology == 'star':
        if center == None:
            center = random.choice(list(range(num_agents)))
            print("Participant {} is the center of the communication network.".format(center + 1))
        for i in range(num_agents):
            if i != center:
                contact.append('participant {}'.format(center + 1))
            else:
                agents.remove(center + 1)

                sent = ""
                for j in agents[:-1]:
                    sent += 'participant {} '.format(j)
                sent += 'and participant {}'.format(agents[-1])
                contact.append(sent)
    elif topology == 'chain':
        contact.append('participant {}'.format(agents[1]))
        for i in range(1, num_agents - 1):
            contact.append('participant {} and participant {}'.format(agents[i-1], agents[i+1]))
        contact.append('participant {}'.format(agents[-2]))
    elif topology == 'complete':
        for i in range(num_agents):
            contact.append("all the other {} participants".format(num_agents-1))
    elif topology == 'tree':
        pass

    return contact, center

def create_agent_config(num_agents, cards, contact):
    agent_config = []
    # cards, answer = create_cards(num_agents, num_signs=num_signs)
    
    # contact = create_topology(num_agents, topology)
    for i in range(num_agents):
        agent_config.append({'no': i+1, 'card': cards[i], 'links': contact[i]})
    return agent_config

def check_answer(msgs, right_answer, agent_num):
    done = [False for _ in range(agent_num)]
    correct = [False for _ in range(agent_num)]
    for msg in msgs:
        if msg.target==-1:
            done[msg.source] = True
            print("Agent {} submit the answer: {}".format(msg.source+1, msg.content))
            if right_answer.lower() in msg.content.lower():
                correct[msg.source] = True
    
    return done, correct


if __name__ == '__main__':
    for i in range(100):
        cards = create_cards(num_agents=6, num_signs=6)
        print(cards)
    # agent_config, ans = create_agent_config(num_agents=6, topology='chain', num_signs=6)
    # print(agent_config)