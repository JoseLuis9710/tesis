#import gym
import numpy as np
from env2 import environment as environment
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import dqn_agent
from tqdm import tqdm
import time


def show_progress(reward,reward2,reward3,reward4,reward5,reward6,reward7,reward8):
    
    
    plt.figure()
    color = ['r','b','g','c']
    line1, = plt.plot(np.arange(len(reward)),reward,color=color[0], label='Surgery 1')
    line2, = plt.plot(np.arange(len(reward2)),reward2,color=color[1], label='Surgery 2')
    line3, = plt.plot(np.arange(len(reward3)),reward3,color=color[2], label='eMBB')
    line4, = plt.plot(np.arange(len(reward4)),reward4,color=color[3], label='MIoT')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.ylabel("Success rate")
    plt.xlabel("Episodes")
    plt.autoscale()
    plt.grid(True)
    plt.show()
    
    # plt.subplot(221)
    # plt.plot(np.arange(len(reward)),reward)
    # #plt.yscale('linear')
    # plt.title('Surgery 1')
    # plt.grid(True)
    
    # # log
    # plt.subplot(222)
    # plt.plot(np.arange(len(reward2)),reward2)
    # #plt.yscale('log')
    # plt.title('Surgery 2')
    # plt.grid(True)
    
    # # symmetric log
    # plt.subplot(223)
    # plt.plot(np.arange(len(reward3)),reward3)
    # #plt.yscale('symlog', linthresh=0.01)
    # plt.title('eMBB')
    # plt.grid(True)
    
    # # logit
    # plt.subplot(224)
    # plt.plot(np.arange(len(reward4)),reward4)
    # #plt.yscale('logit')
    # plt.title('MIoT')
    # plt.grid(True)
    plt.plot(np.arange(len(reward5)),reward5)
    #plt.yscale('linear')
    #plt.title('% Aceptación')
    plt.ylabel("Acceptance rate")
    plt.xlabel("Episodes")
    plt.autoscale()
    plt.grid(True)
    plt.show()
    
    # log
    plt.plot(np.arange(len(reward6)),reward6)
    #plt.yscale('log')
    plt.ylabel('Reward')
    plt.xlabel("Episodes")
    plt.autoscale()
    plt.grid(True)
    plt.show()
    
    # symmetric log
    plt.plot(np.arange(len(reward7)),reward7)
    #plt.yscale('symlog', linthresh=0.01)
    plt.ylabel('Desployment rate')
    plt.xlabel("Episodes")
    plt.autoscale()
    plt.grid(True)
    plt.show()
    
    # logit
    plt.plot(np.arange(len(reward8)),reward8)
    #plt.yscale('logit')
    plt.ylabel('Profit')
    plt.xlabel("Episodes")
    plt.autoscale()
    plt.grid(True)
    plt.show()

sr1= []
sr2= []
emb= []
miot= []
acep_rate= []
rew= []
desp_rate= []
pro_rate= []
#betas = [[0.6,0.2,0.5,0.5],[0.65,0.1,0.25],[0.45,0.45,0.1],[0.15,0.75,0.1],[0.1,0.1,0.8]]
#alphas = [[0.6,0.2,0.1,0.1],[0.2,0.6,0.1,0.1],[0.25,0.25,0.25,0.25],[0.1,0.1,0.6,0.2],[0.4,0.4,0.1,0.1],[0.1,0.1,0.2,0.6],[0.1,0.1,0.4,0.4]]
#descuentos = [0.1,0.3,0.5,0.99]
bch_sz=[32]
#poten =[[0.2,0.8],[0.8,0.2],[0.5,0.5],[0.6,0.4],[0.4,0.6]]
#sigmas =[[250,250]]
#acciones2=[[0.2,0.3,0.7,0.7],[0.2,0.2,0.7,0.5],[0.2,0.2,0.7,0.2],[0.2,0.3,0.7,0.2],[0,0.3,0.7,0.7],[0,0.2,0.7,0.5],[0,0.2,0.7,0.2],[0,0.3,0.7,0.2],[0.2,0.3,0.7,0.2],[0,0.2,0.2,0.2],[0,0.3,0.5,0.5],[0.2,0.2,0.2,0.2],[0,0,0,0],[0,0,0.8,0.8],[0.2,0.3,1,0.2],[0.5,1,1,0.7],[0.2,1,1,1],[0,0,1,1],[0,0,1,0.1],[0,0.1,0.2,0.2],[0,0,0.2,0.2]]
#acciones1=[[0.2,0.3,0.7,0.7],[0.2,0.2,0.7,0.5],[0.2,0.2,0.7,0.2],[0.2,0.3,0.7,0.2],[0,0.3,0.7,0.7],[0,0.2,0.7,0.5],[0,0.2,0.7,0.2],[0,0.3,0.7,0.2],[0.2,0.3,0.7,0.2],[0,0.3,0.7,0],[0.2,0.3,0.7,0],[0,0.2,0.2,0.2],[0,0.3,0.5,0.5],[0.2,0.2,0.2,0.2],[0,0,0,0],[0,0,0.8,0.8],[0.5,0.3,1,0.7],[0.5,0.3,1,0.2],[0.2,0.3,1,0.2],[0.5,1,1,0.7],[0.2,1,1,1],[0,1,1,0],[0,1,1,0.2],[0,0,1,1],[0,0.6,1,1],[0,0,1,0.1],[0.1,1,1,0.5],[0,0.1,0.2,0.2],[0.2,0.5,1,1],[0,0,0.2,0.2],[0.2,0,0.7,0.2]]

for top in bch_sz:

    runs = 1
    reward_total = list()
    reward_total2 = list()
    reward_total3 = list()
    reward_total4 = list()
    reward_total5 = list()
    reward_total6 = list()
    reward_total7 = list()
    reward_total8 = list()
    for _ in tqdm(range(0,runs)):
        env = environment(32,3)
        #env.wg = bet
        #print(bet)
        agente = dqn_agent.Agent(8,32,[250,250])#,replay_start_size=rss)
        env.total()
        
        
        episodes = 1
        episode_rewards = []
        episode_rewards2 = [] 
        episode_rewards3 = []
        episode_rewards4 = []
        episode_rewards5 = []
        episode_rewards6 = [] 
        episode_rewards7 = []
        episode_rewards8 = []
        
        for episode in range(episodes):
            #rint("Episode: {0}".format(episode))
            #print(agente.batch_size)
            steps = 0
            episode_reward = 0
        
            agente.handle_episode_start()    
            s,done = env.env_start()
            #print('ESTADO:',s)
            #print(np.float32(list(s)))
            a = agente.step(np.float32(list(s)),0)
            #print('ACTION:',env.actions[a])
            #s2 = env2.reset()    
            
            while True:
                steps += 1          
                reward, s_ , done = env.env_step(int(a)) 
                episode_reward += reward
                #if episode > 9800:
                    #print('REWARD:',reward)
                    #print("**************************+")
                    #print('ESTADO:',s_)
                #print(env.conteo)
                #print('PETICIONES:',env.pet_llegadas)
                a_ = agente.step(np.float32(list(s_)),reward)
                #if episode > 4800:
                    #print('ACTION:',env.actions[a_])
                    #time.sleep(3)
                s, a = s_ , a_
                #print(steps)
                #random actions
                #a2 = env2.action_space.sample()
                #s2, reward2, done2, info2 = env.step(a2)
                #episode_reward2 += reward2
                
                if steps == 20:
                    done = True
                if done: 
                    if episode % 2950 == 0:
                        show_progress(episode_rewards[2400:],episode_rewards2[2400:],episode_rewards3[2400:],episode_rewards4[2400:],
                                      episode_rewards5[2400:],episode_rewards6[2400:],episode_rewards7[2400:],episode_rewards8[2400:])
                    #print('EPSILON:',agente.max_explore - (agente.steps * agente.anneal_rate))
                    #print('TASA DE ACEPTACIÓN:',env.acept/env.arriv)
                    #episode_rewards.append(episode_reward)
                    episode_rewards6.append(episode_reward/20)
                    episode_rewards.append(sum(env.step_mon)/len(env.step_mon))
                    #episode_rewards.append(env.cir_acep/env.cir)
                    episode_rewards2.append(sum(env.step_mon2)/len(env.step_mon2))
                    #episode_rewards2.append(env.cir2_acep/env.cir2)
                    #episode_rewards3.append(env.emb_acep/env.emb)
                    episode_rewards3.append(sum(env.step_mon_emb)/len(env.step_mon_emb))
                    episode_rewards4.append(sum(env.step_mon_miot)/len(env.step_mon_miot))
                    #episode_rewards4.append(env.miot_acep/env.miot)
                    #episode_rewards5.append(env.acept/env.arriv)
                    if top == 32:
                        episode_rewards5.append(env.multas/446000)
                    if top == 64:
                        episode_rewards5.append(env.multas/910000) # 64 NODOS
                    ss = episode_rewards[-1] + episode_rewards2[-1] + episode_rewards3[-1] + episode_rewards4[-1]
                    episode_rewards7.append(ss/4)
                    if top == 32:
                        episode_rewards8.append(env.sum_profit/446000)
                    if top == 64:
                        episode_rewards8.append(env.sum_profit/910000)# 64 NODOS
                    #episode_rewards8.append(env.U)
                    #episode_rewards.append(episode_reward)
                    #episode_rewards2.append(episode_reward2)
                    break
        
        reward_total.append(episode_rewards)
        reward_total2.append(episode_rewards2)
        reward_total3.append(episode_rewards3)
        reward_total4.append(episode_rewards4)
        reward_total5.append(episode_rewards5)
        reward_total6.append(episode_rewards6)
        reward_total7.append(episode_rewards7)
        reward_total8.append(episode_rewards8)
        
    
    re = np.array(reward_total)
    rem = np.mean(re,axis=0)[2000:]#[7599:8599]
    re2 = np.array(reward_total2)
    rem2 = np.mean(re2,axis=0)[2000:]#[7599:8599]
    re3 = np.array(reward_total3)
    rem3 = np.mean(re3,axis=0)[2000:]#[7599:8599]
    re4 = np.array(reward_total4)
    rem4 = np.mean(re4,axis=0)[2000:]#[7599:8599]
    re5 = np.array(reward_total5)
    rem5 = np.mean(re5,axis=0)[2000:]#[7599:8599]
    re6 = np.array(reward_total6)
    rem6 = np.mean(re6,axis=0)[2000:]#[7599:8599]
    re7 = np.array(reward_total7)
    rem7 = np.mean(re7,axis=0)[2000:]#[7599:8599]
    re8 = np.array(reward_total8)
    rem8 = np.mean(re8,axis=0)[2000:]#[7599:8599]
    
    sr1.append(rem)
    sr2.append(rem2)
    emb.append(rem3)
    miot.append(rem4)
    acep_rate.append(rem5)
    rew.append(rem6)
    desp_rate.append(rem7)
    pro_rate.append(rem8)
    
    

sz = len(sr1);
sr0=np.zeros(sz)

plt.figure()
color = ['r','b','g','c','y','m','k']

#line1, = plt.plot(np.arange(len(sr1[0][470:])),sr1[0][470:],color=color[0], label="32-nodes")
#line2, = plt.plot(np.arange(len(sr1[0][470:])),sr1[0][470:],color=color[1], label="64-nodes")
#line3, = plt.plot(np.arange(len(sr1[2][470:])),sr1[2][470:],color=color[2], label="bs=30")
#line4, = plt.plot(np.arange(len(sr0)),sr0,color='w')
#line4, = plt.plot(np.arange(len(sr1[3][470:])),sr1[3][470:],color=color[3], label="bs=45")
#line5, = plt.plot(np.arange(len(sr1[4][470:])),sr1[4][470:],color=color[4], label="E")
#line6, = plt.plot(np.arange(len(sr1[5][470:])),sr1[5][470:],color=color[5], label="F")
#line7, = plt.plot(np.arange(len(sr1[6][470:])),sr1[6][470:],color=color[6], label="G")
#plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
#plt.title('Surgery 1')
#plt.ylabel("Success rate")
#plt.xlabel("Episodes")
#plt.grid(True)
#plt.ylim(0.2,1)
#plt.savefig('SR1_alphas.eps')
#plt.show()


#plt.figure()
#line1, = plt.plot(np.arange(len(sr2[0][470:])),sr2[0][470:],color=color[0], label="32-nodes")
#line2, = plt.plot(np.arange(len(sr2[0][470:])),sr2[0][470:],color=color[1], label="64-nodes")
#line3, = plt.plot(np.arange(len(sr2[2][470:])),sr2[2][470:],color=color[2], label="bs=30")
#line4, = plt.plot(np.arange(len(sr0)),sr0,color='w')
#line4, = plt.plot(np.arange(len(sr2[3][470:])),sr2[3][470:],color=color[3], label="bs=45")
#line5, = plt.plot(np.arange(len(sr2[4][470:])),sr2[4][470:],color=color[4], label="E")
#line6, = plt.plot(np.arange(len(sr2[5][470:])),sr2[5][470:],color=color[5], label="F")
#line7, = plt.plot(np.arange(len(sr2[6][470:])),sr2[6][470:],color=color[6], label="G")
#plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
#plt.title('Surgery 2')
#plt.ylabel("Success rate")
#plt.xlabel("Episodes")
#plt.grid(True)
#plt.ylim(0.5,1)
#plt.savefig('SR2_alphas.eps')
#plt.show()


#plt.figure()
#line1, = plt.plot(np.arange(len(emb[0][470:])),emb[0][470:],color=color[0], label="32-nodes")
#line2, = plt.plot(np.arange(len(emb[0][470:])),emb[0][470:],color=color[1], label="64-nodes")
#line3, = plt.plot(np.arange(len(emb[2][470:])),emb[2][470:],color=color[2], label="bs=30")
#line4, = plt.plot(np.arange(len(sr0)),sr0,color='w')
#line4, = plt.plot(np.arange(len(emb[3][470:])),emb[3][470:],color=color[3], label="bs=45")
#line5, = plt.plot(np.arange(len(emb[4][470:])),emb[4][470:],color=color[4], label="E")
#line6, = plt.plot(np.arange(len(emb[5][470:])),emb[5][470:],color=color[5], label="F")
#line7, = plt.plot(np.arange(len(emb[6][470:])),emb[6][470:],color=color[6], label="G")
#plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
#plt.title('eMBB')
#plt.ylabel("Success rate")
#plt.xlabel("Episodes")
#plt.grid(True)
#plt.ylim(0.5,1.1)
#plt.savefig('EMBB_alphas.eps')
#plt.show()


#plt.figure()
#line1, = plt.plot(np.arange(len(miot[0][470:])),miot[0][470:],color=color[0], label="32-nodes")
#line2, = plt.plot(np.arange(len(miot[0][470:])),miot[0][470:],color=color[1], label="64-nodes")
#line3, = plt.plot(np.arange(len(miot[2][470:])),miot[2][470:],color=color[2], label="bs=30")
#line4, = plt.plot(np.arange(len(sr0)),sr0,color='w')
#line4, = plt.plot(np.arange(len(miot[3][470:])),miot[3][470:],color=color[3], label="bs=45")
#line5, = plt.plot(np.arange(len(miot[4][470:])),miot[4][470:],color=color[4], label="E")
#line6, = plt.plot(np.arange(len(miot[5][470:])),miot[5][470:],color=color[5], label="F")
#line7, = plt.plot(np.arange(len(miot[6][470:])),miot[6][470:],color=color[6], label="G")
#plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
#plt.title('mMTC')
#plt.ylabel("Success rate")
#plt.xlabel("Episodes")
#plt.grid(True)
#plt.ylim(0.9,1.1)
#plt.savefig('MIOT_alphas.eps')
#plt.show()


#plt.figure()
#line1, = plt.plot(np.arange(len(acep_rate[0][470:])),acep_rate[0][470:],color=color[0], label="32-nodes")
#line2, = plt.plot(np.arange(len(acep_rate[0][470:])),acep_rate[0][470:],color=color[1], label="64-nodes")
#line3, = plt.plot(np.arange(len(acep_rate[2][470:])),acep_rate[2][470:],color=color[2], label="bs=30")
#line4, = plt.plot(np.arange(len(acep_rate[3][470:])),acep_rate[3][470:],color=color[3], label="bs=45")
#line5, = plt.plot(np.arange(len(acep_rate[4][470:])),acep_rate[4][470:],color=color[4], label="E")
#line6, = plt.plot(np.arange(len(acep_rate[5][470:])),acep_rate[5][470:],color=color[5], label="F")
#line7, = plt.plot(np.arange(len(acep_rate[6][470:])),acep_rate[6][470:],color=color[6], label="G")
#plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
#plt.ylabel("Fines")
#plt.xlabel("Episodes")
#plt.grid(True)
#plt.ylim(0,0.7)
#plt.savefig('MULTAS_alphas.eps')
#plt.show()

#plt.figure()
#line1, = plt.plot(np.arange(len(rew[0][470:])),rew[0][470:],color=color[0], label="32-nodes")
#line2, = plt.plot(np.arange(len(rew[0][470:])),rew[0][470:],color=color[1], label="64-nodes")
#line3, = plt.plot(np.arange(len(rew[2][470:])),rew[2][470:],color=color[2], label="bs=30")
#line4, = plt.plot(np.arange(len(sr0)),sr0,color='w')
#line4, = plt.plot(np.arange(len(rew[3][470:])),rew[3][470:],color=color[3], label="bs=45")
#line5, = plt.plot(np.arange(len(rew[4][470:])),rew[4][470:],color=color[4], label="E")
#line6, = plt.plot(np.arange(len(rew[5][470:])),rew[5][470:],color=color[5], label="F")
#line7, = plt.plot(np.arange(len(rew[6][470:])),rew[6][470:],color=color[6], label="G")
#plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
#plt.ylabel("Reward")
#plt.xlabel("Episodes")
#plt.grid(True)
#plt.savefig('REWARD_alphas.eps')
#plt.show()


#plt.figure()
#line1, = plt.plot(np.arange(len(desp_rate[0][470:])),desp_rate[0][470:],color=color[0], label="32-nodes")
#line2, = plt.plot(np.arange(len(desp_rate[0][470:])),desp_rate[0][470:],color=color[1], label="64-nodes")
#line3, = plt.plot(np.arange(len(desp_rate[2][470:])),desp_rate[2][470:],color=color[2], label="bs=30")
#line4, = plt.plot(np.arange(len(sr0)),sr0,color='w')
#line4, = plt.plot(np.arange(len(desp_rate[3][470:])),desp_rate[3][470:],color=color[3], label="bs=45")
#line5, = plt.plot(np.arange(len(desp_rate[4][470:])),desp_rate[4][470:],color=color[4], label="E")
#line6, = plt.plot(np.arange(len(desp_rate[5][470:])),desp_rate[5][470:],color=color[5], label="F")
#line7, = plt.plot(np.arange(len(desp_rate[6][470:])),desp_rate[6][470:],color=color[6], label="G")
#plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
#plt.ylabel("Deployment rate")
#plt.xlabel("Episodes")
#plt.grid(True)
#plt.ylim(0.7,1)
#plt.savefig('DESPLIEGUE_alphas.eps')
#plt.show()

#plt.figure()
#line1, = plt.plot(np.arange(len(pro_rate[0][470:])),pro_rate[0][470:],color=color[0], label="32-nodes")
#line2, = plt.plot(np.arange(len(pro_rate[0][470:])),pro_rate[0][470:],color=color[1], label="64-nodes")
#line3, = plt.plot(np.arange(len(pro_rate[2][470:])),pro_rate[2][470:],color=color[2], label="bs=30")
#line4, = plt.plot(np.arange(len(sr0)),sr0,color='w')
#line4, = plt.plot(np.arange(len(pro_rate[3][470:])),pro_rate[3][470:],color=color[3], label="bs=45")
#line5, = plt.plot(np.arange(len(pro_rate[4][470:])),pro_rate[4][470:],color=color[4], label="E")
#line6, = plt.plot(np.arange(len(pro_rate[5][470:])),pro_rate[5][470:],color=color[5], label="F")
#line7, = plt.plot(np.arange(len(pro_rate[6][470:])),pro_rate[6][470:],color=color[6], label="G")
#line6, = plt.plot(np.arange(len(pro_rate[5][470:])),pro_rate[5][470:],color=color[5], label="F")
#plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
#plt.ylabel("Profit")
#plt.xlabel("Episodes")
#plt.grid(True)
#plt.ylim(0.1,0.65)
#plt.savefig('profit_alphas.eps')
#plt.show()