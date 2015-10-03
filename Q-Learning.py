import random
from operator import add
from operator import mul
import numpy as np
import pandas as pd
import csv

class QLearn:
    
    def __init__(self, actions, epsilon=0.3, alpha=0.2, gamma=0.6):
        self.q = {}

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)
        # return self.q.get((state, action), 1.0)

    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)


    def getState(self,state,action):
    	#print "---->",state
    	def div(x): return float(x/10);
        def rnd(x): return round(x,2);
        old_state=state
        

        state = map(float,state)
        #print "xxx",state
        state = map(div,state)
        #print state
        action=int(action)
        acts = {0:[0.2,0,-0.2], 1:[-0.2,0,0.2], 2:[0.2,-0.2,0], 3:[0,0,0], 4:[-0.2,0.2,0], 5:[0,0.2,-0.2], 6:[0,-0.2,0.2] }
        print "state-->",state,"action-->",acts[action]
        state = map(add,state,acts[action])
        state= map(rnd,state)
        print "newState",state
        if any(t < 0 for t in state):
        	return -1
        def multi(x): return int(10*x)
        state=map(multi,state)
        state=map(str,state)
        state="".join(state)
        return state


    def getReward(self,state,userValue,parameters):
    	def multi(x,y):
    		return x*y
    	def div(x):return x/10;
    	state=map(float,state)
    	state=map(div,state)
    	parameters=map(float,parameters)
    	#print "state",state
        predictedWeight = sum(map(multi,state,parameters))
        print "predictedWeight",predictedWeight
        error = abs(predictedWeight - userValue)
        reward = 1/(error + 1)
        minimum = 0.17
        maximum = 1
        num = reward - minimum
        den = maximum - minimum
        return 5*num/den 


    def chooseAction(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            print "random"
        else:
        
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            print "informed"
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

    def printQ(self):
        keys = self.q.keys()
        states = list(set([a for a,b in keys]))
        actions = list(set([b for a,b in keys]))
        
        dstates = ["".join([str(int(t)) for t in list(tup)]) for tup in states]
        print (" "*4) + " ".join(["%8s" % ("("+s+")") for s in dstates])
        for a in actions:
            print ("%3d " % (a)) + \
                " ".join(["%8.2f" % (self.getQ(s,a)) for s in states])

    def printV(self):
        keys = self.q.keys()
        states = [a for a,b in keys]
        statesX = list(set([x for x,y in states]))
        statesY = list(set([y for x,y in states]))

        print (" "*4) + " ".join(["%4d" % (s) for s in statesX])
        for y in statesY:
            maxQ = [max([self.getQ((x,y),a) for a in self.actions])
                    for x in statesX]
            print ("%3d " % (y)) + " ".join([ff(q,4) for q in maxQ])
        
import math
def ff(f,n):
    fs = "{:f}".format(f)
    if len(fs) < n:
        return ("{:"+n+"s}").format(fs)
    else:
        return fs[:n]
    # s = -1 if f < 0 else 1
    # ss = "-" if s < 0 else ""
    # b = math.floor(math.log10(s*f)) + 1
    # if b >= n:
    #     return ("{:" + n + "d}").format(math.round(f))
    # elif b <= 0:
    #     return (ss + ".{:" + (n-1) + "d}").format(math.round(f * 10**(n-1)))
    # else:
    #     return ("{:"+b+"d}.{:"+(n-b-1)+"



qLearn = np.array(list(csv.reader(open("Reinforcement.csv","rb"),delimiter=',')))
parameter_rating = np.array(list(csv.reader(open("parameter.csv","rb"),delimiter=',')))
states = qLearn[0,1:]
state = '163'
actions = range(0,7)
user = QLearn(actions)
cnt=1
acts = {0:[0.2,0,-0.2], 1:[-0.2,0,0.2], 2:[0.2,-0.2,0], 3:[0,0,0], 4:[-0.2,0.2,0], 5:[0,0.2,-0.2], 6:[0,-0.2,0.2] }
for i in xrange(0,9998):
	cnt=cnt+1
	print "state",state
	action = int(user.chooseAction(state))
	print "action",acts[action]
	newState = user.getState(state,action)

	if newState ==-1:
		i=i-1
		continue;

	parameters=parameter_rating[cnt,0:3]
	print "parameters",parameters
	userValue=float(parameter_rating[cnt,3])
	print "userValue",userValue
	reward = float(user.getReward(newState,userValue,parameters))
	print "reward",reward

	user.learn(state,action,reward,newState)
	def multi(x,y):
		return x*y
	def div(x):return x/10;
	state_d=newState
	state_d = map(float,state_d)
	state_d = map(div,state_d)
	#print "state_d",state_d
	parameters=map(float,parameters)
	Predicted_value=sum(map(multi,state_d,parameters))
	print "Predicted_value",Predicted_value,"Actual_value",userValue,"Difference",Predicted_value-userValue
	print "newState",newState
	state = newState
	print "--------------------------------------------------------------------------------------------------------------"
	#user.printQ()
	
#marks = np.array(list(csv.reader(open("marks.csv","rb"),delimiter=',', header = 'true')))
# marks = pd.read_csv('marks.csv', skiprows=1)
# print marks.iloc[[1]]

# csv = pd.read_csv('LOSkill.csv')
# print csv
# print csv[2]

LOSkill = np.loadtxt('LOSkill.csv',delimiter = ',',skiprows = 1)
marks = np.loadtxt('marks.csv',delimiter = ',',dtype = str, skiprows = 1)
userSkill =  np.loadtxt('UserSkillLevel.csv',delimiter = ',', dtype = None, skiprows = 1)

#print marks

marksDict = {}
userSkillDict = {}

for i in xrange(0,len(marks)):
    marksDict[marks[i][1]] = {}
    userSkillDict[marks[i][1]] = {}
    for j in xrange(2,len(marks[0])):
        marksDict[marks[i][1]]['skill1'] = marks[i][2]
        marksDict[marks[i][1]]['skill2'] = marks[i][3]
        marksDict[marks[i][1]]['skill3'] = marks[i][4]
        marksDict[marks[i][1]]['skill4'] = marks[i][5]
        marksDict[marks[i][1]]['skill5(a)'] = marks[i][6]
        marksDict[marks[i][1]]['skill5(b)'] = marks[i][7]
        marksDict[marks[i][1]]['skill6'] = marks[i][8]
        marksDict[marks[i][1]]['skill7(a)'] = marks[i][9]
        marksDict[marks[i][1]]['skill7(b)'] = marks[i][10]

        userSkillDict[marks[i][1]]['skill1'] = userSkill[i][1]
        userSkillDict[marks[i][1]]['skill2'] = userSkill[i][2]
        userSkillDict[marks[i][1]]['skill3'] = userSkill[i][3]
        userSkillDict[marks[i][1]]['skill4'] = userSkill[i][4]
        userSkillDict[marks[i][1]]['skill5(a)'] = userSkill[i][5]
        userSkillDict[marks[i][1]]['skill5(b)'] = userSkill[i][6]
        userSkillDict[marks[i][1]]['skill6'] = userSkill[i][7]
        userSkillDict[marks[i][1]]['skill7(a)'] = userSkill[i][8]
        userSkillDict[marks[i][1]]['skill7(b)'] = userSkill[i][9]
        
# print marksDict

#print userSkillDict

userID = '13BCE0410'


print userSkillDict[userID]


for skill,Marks in marksDict[userID].items():
    if float(Marks) < 5 and float(userSkillDict[userID][skill]) != 0:
        userSkillDict[userID][skill] = float(userSkillDict[userID][skill]) - 1
    elif float(Marks) > 7:
        userSkillDict[userID][skill] = float(userSkillDict[userID][skill]) + 1


print userSkillDict[userID]


