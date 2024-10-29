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

def run_one_round(manager: AgentDialogManagement, answer, save_path, args, max_length=30):
    done = False
    correct = False
    
    msgs = manager.initiate_chat()

    agent_num = len(msgs)
    cnt = [0 for _ in range(agent_num)]
    done_agents = [False for _ in range(agent_num)]
    correct_agents = [False for _ in range(agent_num)]
    done_all = all(done_agents)
    for msg in msgs:
        print(f"from: {msg.source + 1}, to: {msg.target + 1}, content: {msg.content} \n")
    while not done_all and max(cnt) <= max_length:
                
        done, correct = check_answer(msgs, answer, agent_num)
        # print(done, correct)
        done_agents = [x or y for x, y in zip(done, done_agents)]
        correct_agents = [x or y for x, y in zip(correct_agents, correct)]
        print(done_agents, correct_agents)
        done_all = all(done_agents)

        reply_buffer = []
        if not done_all:
            for i in range(agent_num):
                if not done_agents[i]:
                    cnt[i] += 1
            for msg in msgs:
                # ignore msg that is submission
                if msg.target == -1:
                    continue
                is_inquiry, summary = manager.assess(msg, ASSESS_PROMPT, SUMMARIZE_PROMPT)
                # print(assess, summary)
                # reply = manager.answer_phase()
                evaluation = manager.evaluate(EVALUATE_PROMPT, msg.target)
                reply, response = manager.planning(PLANNING_PROMPT, msg, evaluation, is_evaluation=True, is_inquiry=False)
                manager.add_to_history(msg.target, msg, response)
                reply_buffer.append(reply)
                print(f"from: {reply.source + 1}, to: {reply.target + 1}, content: {reply.content} \n")
                print("--------------------------------------")
            
            msgs = reply_buffer
            if len(msgs) == 0:
                for i in range(agent_num):
                    if not done_agents[i]:
                        msg = manager.ask(ASK_PROMPT, i)
                        msgs.append(msg)

    correct_all = correct_agents.count(True) >= correct_agents.count(False)
    
    return cnt, correct_all, correct_agents

def run_one_round_without_inference(manager: AgentDialogManagement, answer, save_path, args, max_length=30):
    done = False
    correct = False
    
    msgs = manager.initiate_chat()

    agent_num = len(msgs)
    cnt = [0 for _ in range(agent_num)]
    done_agents = [False for _ in range(agent_num)]
    correct_agents = [False for _ in range(agent_num)]
    done_all = all(done_agents)
    for msg in msgs:
        print(f"from: {msg.source + 1}, to: {msg.target + 1}, content: {msg.content} \n")
    while not done_all and max(cnt) <= max_length:
                
        done, correct = check_answer(msgs, answer, agent_num)
        # print(done, correct)
        done_agents = [x or y for x, y in zip(done, done_agents)]
        correct_agents = [x or y for x, y in zip(correct_agents, correct)]
        print(done_agents, correct_agents)
        done_all = all(done_agents)

        reply_buffer = []
        if not done_all:
            for i in range(agent_num):
                if not done_agents[i]:
                    cnt[i] += 1
            for msg in msgs:
                # ignore msg that is submission
                if msg.target == -1:
                    continue
                # is_inquiry, summary = manager.assess(msg, ASSESS_PROMPT, SUMMARIZE_PROMPT)
                # print(assess, summary)
                # reply = manager.answer_phase()
                evaluation = manager.evaluate(EVALUATE_PROMPT_WITHOUT_INFERENCE, msg.target)
                reply, response = manager.planning(PLANNING_PROMPT, msg, evaluation, is_evaluation=True, is_inquiry=False)
                manager.add_to_history(msg.target, msg, response)
                reply_buffer.append(reply)
                print(f"from: {reply.source + 1}, to: {reply.target + 1}, content: {reply.content} \n")
                print("--------------------------------------")
            
            msgs = reply_buffer
            if len(msgs) == 0:
                for i in range(agent_num):
                    if not done_agents[i]:
                        msg = manager.ask(ASK_PROMPT, i)
                        msgs.append(msg)

    correct_all = correct_agents.count(True) >= correct_agents.count(False)
    
    return cnt, correct_all, correct_agents

def run_one_round_without_evaluation(manager: AgentDialogManagement, answer, save_path, args, max_length=30):
    done = False
    correct = False
    
    msgs = manager.initiate_chat()

    agent_num = len(msgs)
    cnt = [0 for _ in range(agent_num)]
    done_agents = [False for _ in range(agent_num)]
    correct_agents = [False for _ in range(agent_num)]
    done_all = all(done_agents)
    for msg in msgs:
        print(f"from: {msg.source + 1}, to: {msg.target + 1}, content: {msg.content} \n")
    while not done_all and max(cnt) <= max_length:
                
        done, correct = check_answer(msgs, answer, agent_num)
        # print(done, correct)
        done_agents = [x or y for x, y in zip(done, done_agents)]
        correct_agents = [x or y for x, y in zip(correct_agents, correct)]
        print(done_agents, correct_agents)
        done_all = all(done_agents)

        reply_buffer = []
        if not done_all:
            for i in range(agent_num):
                if not done_agents[i]:
                    cnt[i] += 1
            for msg in msgs:
                # ignore msg that is submission
                if msg.target == -1:
                    continue
                is_inquiry, summary = manager.assess(msg, ASSESS_PROMPT, SUMMARIZE_PROMPT)
                # print(assess, summary)
                # reply = manager.answer_phase()
                # evaluation = manager.evaluate(EVALUATE_PROMPT, msg.target)
                reply, response = manager.planning(PLANNING_PROMPT, msg, evaluation="", is_evaluation=False, is_inquiry=is_inquiry)
                manager.add_to_history(msg.target, msg, response)
                reply_buffer.append(reply)
                print(f"from: {reply.source + 1}, to: {reply.target + 1}, content: {reply.content} \n")
                print("--------------------------------------")
            
            msgs = reply_buffer
            if len(msgs) == 0:
                for i in range(agent_num):
                    if not done_agents[i]:
                        msg = manager.ask(ASK_PROMPT, i)
                        msgs.append(msg)

    correct_all = correct_agents.count(True) >= correct_agents.count(False)
    
    return cnt, correct_all, correct_agents

def run_one_round_with_plain_cot(manager: AgentDialogManagement, answer, save_path, args, max_length=30):
    done = False
    correct = False
    
    msgs = manager.initiate_chat()

    agent_num = len(msgs)
    cnt = [0 for _ in range(agent_num)]
    done_agents = [False for _ in range(agent_num)]
    correct_agents = [False for _ in range(agent_num)]
    done_all = all(done_agents)
    for msg in msgs:
        print(f"from: {msg.source + 1}, to: {msg.target + 1}, content: {msg.content} \n")
    while not done_all and max(cnt) <= max_length:
                
        done, correct = check_answer(msgs, answer, agent_num)
        # print(done, correct)
        done_agents = [x or y for x, y in zip(done, done_agents)]
        correct_agents = [x or y for x, y in zip(correct_agents, correct)]
        print(done_agents, correct_agents)
        done_all = all(done_agents)

        reply_buffer = []
        if not done_all:
            for i in range(agent_num):
                if not done_agents[i]:
                    cnt[i] += 1
            for msg in msgs:
                # ignore msg that is submission
                if msg.target == -1:
                    continue
                # is_inquiry, summary = manager.assess(msg, ASSESS_PROMPT, SUMMARIZE_PROMPT)
                # print(assess, summary)
                # reply = manager.answer_phase()
                # evaluation = manager.evaluate(EVALUATE_PROMPT, msg.target)
                reply, response = manager.planning(COT_PLANNING_PROMPT, msg, evaluation="", is_evaluation=False, is_inquiry=False, with_inference=False)
                manager.add_to_history(msg.target, msg, response)
                reply_buffer.append(reply)
                print(f"from: {reply.source + 1}, to: {reply.target + 1}, content: {reply.content} \n")
                print("--------------------------------------")
            
            msgs = reply_buffer
            if len(msgs) == 0:
                for i in range(agent_num):
                    if not done_agents[i]:
                        msg = manager.ask(ASK_PROMPT, i)
                        msgs.append(msg)

    correct_all = correct_agents.count(True) >= correct_agents.count(False)
    
    return cnt, correct_all, correct_agents, correct_agents

def run_one_round_with_plain_prompt(manager: AgentDialogManagement, answer, save_path, args, max_length=30):
    done = False
    correct = False
    
    msgs = manager.initiate_chat()

    agent_num = len(msgs)
    cnt = [0 for _ in range(agent_num)]
    done_agents = [False for _ in range(agent_num)]
    correct_agents = [False for _ in range(agent_num)]
    done_all = all(done_agents)
    for msg in msgs:
        print(f"from: {msg.source + 1}, to: {msg.target + 1}, content: {msg.content} \n")
    while not done_all and max(cnt) <= max_length:
                
        done, correct = check_answer(msgs, answer, agent_num)
        # print(done, correct)
        done_agents = [x or y for x, y in zip(done, done_agents)]
        correct_agents = [x or y for x, y in zip(correct_agents, correct)]
        print(done_agents, correct_agents)
        done_all = all(done_agents)

        reply_buffer = []
        if not done_all:
            for i in range(agent_num):
                if not done_agents[i]:
                    cnt[i] += 1
            for msg in msgs:
                # ignore msg that is submission
                if msg.target == -1:
                    continue
                # is_inquiry, summary = manager.assess(msg, ASSESS_PROMPT, SUMMARIZE_PROMPT)
                # print(assess, summary)
                # reply = manager.answer_phase()
                # evaluation = manager.evaluate(EVALUATE_PROMPT, msg.target)
                reply, response = manager.planning(SIMPLE_PLANNING_PROMPT, msg, evaluation="", is_evaluation=False, is_inquiry=False, with_inference=False)
                manager.add_to_history(msg.target, msg, response)
                reply_buffer.append(reply)
                print(f"from: {reply.source + 1}, to: {reply.target + 1}, content: {reply.content} \n")
                print("--------------------------------------")
            
            msgs = reply_buffer
            if len(msgs) == 0:
                for i in range(agent_num):
                    if not done_agents[i]:
                        msg = manager.ask(ASK_PROMPT, i)
                        msgs.append(msg)

    correct_all = correct_agents.count(True) >= correct_agents.count(False)
    
    return cnt, correct_all, correct_agents

def run_one_round_with_self_consistency(manager: AgentDialogManagement, answer, save_path, args, max_length=30):
    done = False
    correct = False
    
    msgs = manager.initiate_chat()

    agent_num = len(msgs)
    cnt = [0 for _ in range(agent_num)]
    done_agents = [False for _ in range(agent_num)]
    correct_agents = [False for _ in range(agent_num)]
    done_all = all(done_agents)
    for msg in msgs:
        print(f"from: {msg.source + 1}, to: {msg.target + 1}, content: {msg.content} \n")
    while not done_all and max(cnt) <= max_length:
                
        done, correct = check_answer(msgs, answer, agent_num)
        # print(done, correct)
        done_agents = [x or y for x, y in zip(done, done_agents)]
        correct_agents = [x or y for x, y in zip(correct_agents, correct)]
        print(done_agents, correct_agents)
        done_all = all(done_agents)

        reply_buffer = []
        if not done_all:
            for i in range(agent_num):
                if not done_agents[i]:
                    cnt[i] += 1
            for msg in msgs:
                # ignore msg that is submission
                if msg.target == -1:
                    continue
                # is_inquiry, summary = manager.assess(msg, ASSESS_PROMPT, SUMMARIZE_PROMPT)
                # print(assess, summary)
                # reply = manager.answer_phase()
                # evaluation = manager.evaluate(EVALUATE_PROMPT, msg.target)
                # print("self consistency planning")
                reply, response = manager.self_consistency_planning(SIMPLE_CONSISTENT_PLANNING_PROMPT, SELF_CONSISTENCY_PROMPT, msg)
                manager.add_to_history(msg.target, msg, response)
                reply_buffer.append(reply)
                print(f"from: {reply.source + 1}, to: {reply.target + 1}, content: {reply.content} \n")
                print("--------------------------------------")
            
            msgs = reply_buffer
            if len(msgs) == 0:
                for i in range(agent_num):
                    if not done_agents[i]:
                        msg = manager.ask(ASK_PROMPT, i)
                        msgs.append(msg)

    correct_all = correct_agents.count(True) >= correct_agents.count(False)
    
    return cnt, correct_all, correct_agents