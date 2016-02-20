from hmm import HMM

alphabet = [0, 1]
states = ['F', 'B']

a = {
   'F': {'F': 0.9, 'B': 0.1},
   'B': {'F': 0.1, 'B': 0.9}
   }
b = {
   'F': { 0: 0.5, 1: 0.5},
   'B': { 0: 0.25, 1: 0.75}
   }
pi = {'F': 0.5, 'B': 0.5}


### arbitrary initial learning HMM params
a2 = {
   'F': {'F': 0.95, 'B': 0.05},
   'B': {'F': 0.05, 'B': 0.95}
   }
b2 = {
   'F': { 0: 0.6, 1: 0.4},
   'B': { 0: 0.35, 1: 0.65}
   }
pi2 = {'F': 0.5, 'B': 0.5}

   
hmm = HMM(a, b, pi)

# Viterbi Test - It Works!!!
# obs = hmm.generate(100, True)
# (prob, path) = hmm.viterbi(obs)
# print prob
# for i in range(len(obs)):
#    print str(obs[i])+' '+path[i]


# Learning Test
hmmLearn = HMM(a2, b2, pi2)
#Defines the influence that new observations have on previous probabilities
hmmLearn.influence = (3, 14)
# hmmLearn.generateMatrix(states, alphabet)


##Supervised Learning
# for x in range(50):
#    hmmLearn.supertrain(hmm.generate(2000))
#    print "-----"
#    print hmmLearn.pi
#    print hmmLearn.a
#    print hmmLearn.b

##Unsupervised Learning (EM/Baum-Welch)
for x in range(100):
   hmmLearn.baum_welch(hmm.generate(100, True))

print pi
print a
print b
print "---"
print hmmLearn.pi
print hmmLearn.a
print hmmLearn.b
