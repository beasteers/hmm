from hmm import HMM

alphabet = [0, 1]
states = ['F', 'B']

a = {
   'F': {'F': 0.9, 'B': 0.1},
   'B': {'F': 0.1, 'B': 0.9}
   }
e = {
   'F': { 0: 0.5, 1: 0.5},
   'B': { 0: 0.25, 1: 0.75}
   }
pi = {'F': 0.5, 'B': 0.5}


###Learning HMM params
a2 = {
   'F': {'F': 0.6, 'B': 0.4},
   'B': {'F': 0.4, 'B': 0.6}
   }
e2 = {
   'F': { 0: 0.2, 1: 0.8},
   'B': { 0: 0.7, 1: 0.3}
   }
pi2 = {'F': 0.5, 'B': 0.5}

   
hmm = HMM(a, e, pi)

# # Viterbi Test - It Works!!!
# obs = hmm.generate(50)
# (prob, path) = hmm.viterbi(obs)
# print prob
# for i in range(len(obs)):
#    print str(obs[i])+' '+path[i]


# Baum-Welch Test
hmmLearn = HMM(a2, e2, pi2)

# hmmLearn.generateMatrix(states, alphabet)
for x in range(50):
   hmmLearn.supertrain(hmm.generate(2000))
   print "-----"
   print hmmLearn.pi
   print hmmLearn.a
   print hmmLearn.b

# for x in range(50):
#    hmmLearn.baum_welch(hmm.generate(200))

# print hmmLearn.pi
# print hmmLearn.a
# print hmmLearn.b
