from openai import OpenAI
import time
import pickle
import os
import numpy as np
import random
import argparse

from prompt import *
import json
from utils.setting import *
from utils.management import *
from run_baseline import *


def main(args):
    save_path = args.output_path + '_{}_agents_{}'.format(args.n_agents, args.topology)
    os.mkdir(save_path)
    manager = AgentDialogManagement(num_agents=args.n_agents, model=args.model, API_KEY=OPENAI_API, SYSTEM_PROMPT=SYSTEM_PROMPT_FORMAT, output_path=save_path)
    
    round_number = args.round
    total_correct = 0
    total_round = 0
    # five_round_acc = 0
    each_agent_acc = [0 for _ in range(args.n_agents)]
    each_agent_round = [0 for _ in range(args.n_agents)]
    f = open(save_path + '/result.txt', 'w')
    contact, center = create_topology(num_agents=args.n_agents, topology=args.topology)
    for i in range(round_number):
        try:
            cards, answer = create_cards(num_agents=args.n_agents, num_signs=args.n_signs)
        except IndexError:
            cards, answer = create_cards(num_agents=args.n_agents, num_signs=args.n_signs)
        agent_config = create_agent_config(num_agents=args.n_agents, cards=cards, contact=contact)
        manager.generate_agents(agent_config, INITIATE_PROMPT, IDENTITY_FORMAT, args.n_signs)

        print("-------Round {} Begin. The correct answer is the {}.--------".format(i, answer))
        if args.mode == 'full':
            cnt, correct, correct_agents = run_one_round(manager, answer, save_path, args)
        elif args.mode == 'w/o eval':
            cnt, correct = run_one_round_without_evaluation(manager, answer, save_path, args)
        elif args.mode == 'w/o infer':
            cnt, correct, correct_agents = run_one_round_without_inference(manager, answer, save_path, args)
        elif args.mode == 'cot':
            cnt, correct, correct_agents = run_one_round_with_plain_cot(manager, answer, save_path, args)
        elif args.mode == 'selfconsis':
            cnt, correct = run_one_round_with_self_consistency(manager, answer, save_path, args)
        else:
            cnt, correct, correct_agents = run_one_round_with_plain_prompt(manager, answer, save_path, args)
        if args.reflect:
            if i % 5 == 0:
                reflection = manager.reflect(REFLECT_PROMPT, correct, answer)
                guess_acc = 0
                for x in reflection:
                    print(x)
                    if args.topology in x:
                        guess_acc += 1
                print("The topology guess accuracy is {}/{} in round {}.\n".format(guess_acc, len(reflection), i))
                f.write("The topology guess accuracy is {}/{} in round {}.\n".format(guess_acc, len(reflection), i))
    
        f.write("Round {} altogether {} rounds.\n".format(i, cnt))
        f.write("Round {} agents answer is {}".format(i, correct_agents))
        file_path = save_path + '/round{}'.format(i)
        manager.save(file_path)
        total_correct += correct
        # five_round_acc += correct
        # if (i + 1) % 5 == 0:
        #     f.write("The 5-round accuracy is {}/{} in round {}.\n".format(five_round_acc, 5, i+1))
        #     five_round_acc = 0
        each_agent_acc = [x + y for x, y in zip(each_agent_acc, correct_agents)]
        each_agent_round = [x + y for x, y in zip(each_agent_round, cnt)]
        total_round += max(cnt)
        print("Round {} altogether {} rounds".format(i, cnt))
        manager.clear_history()

    print("The each agent accuracy is {}".format(each_agent_acc))
    print("Each agent round is {}".format(each_agent_round))
    f.write("The each agent accuracy is {}\n".format(each_agent_acc))
    f.write("Each agent round is {}\n".format(each_agent_round))
    if args.topology == 'star':
        peripheral_acc = (sum(each_agent_acc)-each_agent_acc[center]) / (total_round*(args.n_agents-1))
        print("The peripheral agent accuracy is {}".format(peripheral_acc))
        f.write("The central agent accuracy is {}.\n".format(each_agent_acc[center]/total_round))
    
        
    print("The accuracy is {}/{}".format(total_correct, round_number))
    
    f.write("The accuracy is {}/{}.\n".format(total_correct, round_number))
    f.write("The total rounds of communication is {} round in {} rounds of experiment".format(total_round, round_number))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='gpt-4o')
    parser.add_argument('-a', '--api')
    parser.add_argument('-n', '--n_agents', type=int, default=3)
    parser.add_argument('-s', '--n_signs', type=int, default=3)
    parser.add_argument('-t', '--topology', choices=['circle', 'chain', 'star', 'complete'], default='chain')
    now = time.strftime("%m_%d_%H_%M", time.localtime(time.time()))
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')
    save_path = './experiments/{}'.format(now)
    parser.add_argument('-o', '--output_path', default=save_path)
    parser.add_argument('-r', '--round', type=int, default=1)
    parser.add_argument('-M', '--mode', choices=['w/o eval', 'w/o infer', 'cot', 'plain', 'full', 'selfconsis'], default='full')
    parser.add_argument('-R', '--reflect', type=bool, default=True)
    args = parser.parse_args()
    main(args)