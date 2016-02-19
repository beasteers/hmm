from sets import Set
import random

# observations = ['normal', 'cold', 'dizzy', 'cold', 'cold', 'dizzy', 'normal', 'normal']
# observations2 = ['normal', 'normal', 'dizzy', 'dizzy', 'cold', 'dizzy', 'normal', 'normal', 'dizzy', 'normal', 'normal', 'normal', 'dizzy', 'cold', 'normal', 'normal', 'dizzy', 'cold', 'normal', 'normal', 'dizzy']
#  
# start_probability = {'Healthy': 0.6, 'Fever': 0.4}
#  
# transition_probability = {
#    'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
#    'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
#    }
#  
# emission_probability = {
#    'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
#    'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
#    }


class HMM:
   def __init__(self, a=False, b=False, pi=False):
      self.hiddenStates = []
      self.observableStates = []
      
      self.a = {}
      self.b = {}
      self.pi = {}
      if a and b and pi:
         self.defHMM(a, b, pi)
      
      self.alpha = [] #forward probabilities
      self.beta = [] #backward probabilities
      
      self.delta = []
      self.path = {}
      
      self.obs = []
      
      self.influence = (1, 10)
      
   
   def generateMatrix(self, hidden, observable):
      self.hiddenStates = hidden
      self.observableStates = observable
      for i in hidden:
         self.a[i] = {}
         self.b[i] = {}
         self.pi[i] = 0.5
         for e in observable:
            self.b[i][e] = 0.5
         for j in hidden:
            self.a[i][j] = 0.5
            
   def defHMM(self, a, b, pi):
      for hidden in b:
         self.hiddenStates.append(hidden)
         self.observableStates.extend(b[hidden].keys())
      #remove duplicates
      self.observableStates = list(Set(self.observableStates))
      
      self.a = a
      self.b = b
      self.pi = pi
   
   def generate(self, length):
      seq = []
      n = self.weightedRandom(self.pi)
      seq.append(  (n, self.weightedRandom(self.b[n]))  )
      for l in range(length):
         n = self.weightedRandom(self.a[n])
         #seq.append(self.weightedRandom(self.b[n]))
         seq.append(  (n, self.weightedRandom(self.b[n]))  )
      return seq
         

   def weightedRandom(self, arr):
      rnd = random.random()
      for i, w in arr.iteritems():
         rnd -= w
         if rnd < 0:
            return i
   
   ## Decoding:
   ## Most probable path w/ relative probability
   ##############################################
   def viterbi(self, obs = False):
      ##If no observation, just return the last value
      if obs:
         ## If alpha isn't set, initialize
         start = 0
         if len(self.delta) == 0:
            start = 1
            initdelta = {}
            for j in self.hiddenStates:
               initdelta[j] = self.pi[j] * self.b[j][obs[0]]
               self.path[j] = [j]
            self.delta.append(initdelta)
         
         for t in range(start, len(obs)):
            lastdelta = self.delta[-1]
            delta = {}
            newpath = {}
            # loop through each of the states the current t
            for j in self.hiddenStates:
               delta[j] = 0
               #max probabilities at that state
               (prob, state) = max((lastdelta[i] * self.a[i][j], i) for i in self.hiddenStates)
               newpath[j] = self.path[state] + [j]
               delta[j] = self.b[j][obs[t]] * prob
            self.path = newpath  
            self.delta.append(delta)
      
      #Return Value
      (prob, state) = max((self.delta[-1][i], i) for i in self.hiddenStates)
      # prob /= sum(self.delta[-1][i] for i in self.hiddenStates)
      path = self.path[state]
      # for x in range(len(obs)):
      #    print obs[x]+': '+path[x]
         
      return (prob, path)
   
   
   def forward(self, obs = False):
      ##If no observation, just return the last value
      if obs:
         start = 0
         self.alpha = [] #I realize this makes the next conditional useless
         ## If alpha isn't set, initialize
         if len(self.alpha) == 0:
            start = 1
            initalpha = {}
            for j in self.hiddenStates:
               initalpha[j] = self.pi[j] * self.b[j][obs[0]]
            self.alpha.append(initalpha)
         
         for t in range(start, len(obs)):
            lastalpha = self.alpha[-1]
            alpha = {}
            # loop through each of the states the current t
            for j in self.hiddenStates:
               alpha[j] = 0
               #sum probabilities to that state
               for i in lastalpha:
                  alpha[j] += lastalpha[i] * self.a[i][j]
               alpha[j] *= self.b[j][obs[t]]
               
            self.alpha.append(alpha)
      
      #Return Value
      Pr = 0
      for j in self.hiddenStates:
         Pr += self.alpha[-1][j]
      return Pr
      
   
   def backward(self, obs=False):
      ##If no observation, just return the last value
      if obs:
         self.beta = []
         ## If beta isn't set, initialize
         if len(self.beta) == 0:
            initbeta = {}
            for j in self.hiddenStates:
               initbeta[j] = 1
            self.beta.append(initbeta)
      
         for t in reversed(xrange(len(obs)-1)):
            lastbeta = self.beta[0]
            beta = {}
            for i in self.hiddenStates:
               beta[i] = 0
               for j in lastbeta:
                  beta[i] += lastbeta[j] * self.a[i][j] * self.b[j][obs[t+1]]
            self.beta.insert(0, beta)
      Pr = 0
      for j in self.hiddenStates:
         Pr += self.beta[t][j]
      return Pr
   
   
   def baum_welch(self, obs):
      self.obs = obs
      self.forward(obs)
      self.backward(obs)

      ######
      
      T = len(obs)
      for i in self.hiddenStates:
         self.pi[i] = self._gamma(0, i)
      
      for v in self.observableStates:
         for i in self.hiddenStates:
            gamma_sum = 0
            gamma_match = 0
            for t in range(T):
               gamma_i = self._gamma(t, i)
               if obs[t] == v:
                  gamma_match += gamma_i
               gamma_sum += gamma_i
            self.b[i][v] = gamma_match / gamma_sum
      
      for i in self.hiddenStates:
         gamma_sum = 0
         for t in range(T-1):
            gamma_sum += self._gamma(t, i)
         for j in self.hiddenStates:
            zeta_sum = 0
            for t in range(T-1):
               zeta_i = self._zeta(t, i, j)
               zeta_sum += zeta_i
            self.a[i][j] = zeta_sum / gamma_sum
      
   def _gamma(self, t, i):
      den = 0;
      for j in self.hiddenStates:
         den += self.alpha[t][j] * self.beta[t][j]
      # print str(self.alpha[t][i] * self.beta[t][i])+" "+str(den)+" "+str(self.alpha[t][i] * self.beta[t][i]/den)
      return self.alpha[t][i] * self.beta[t][i] / den
   
   def _zeta(self, t, i, j):
      den = 0
      for k in self.hiddenStates:
         for l in self.hiddenStates:
            den += self.alpha[t][k] * self.a[k][l] * self.beta[t+1][l] * self.b[l][self.obs[t+1]]
      return self.alpha[t][i] * self.a[i][j] * self.beta[t+1][j] * self.b[j][self.obs[t+1]] / den
   
   
   def supertrain(self, seq):
      a = {}
      b = {}
      pi = {}
      for h in self.hiddenStates:
         a[h] = {}
         b[h] = {}
         pi[h] = 0
      lastState = False
      for s in seq:
         (state, obs) = s
         ##a
         if lastState:
            if state not in a[lastState]: a[lastState][state] = 0
            a[lastState][state] += 1
         else:
            pi[state] = 1
         lastState = state
         
         ##b
         if obs not in b[state]: b[state][obs] = 0
         b[state][obs] += 1
      
      for i in self.hiddenStates:
         a[i] = self._normalize(a[i])
         self.a[i] = self._influence(self.a[i], a[i])
         b[i] = self._normalize(b[i])
         self.b[i] = self._influence(self.b[i], b[i])
      pi = self._normalize(pi)
      self.pi = self._influence(self.pi, pi)

      
   def _normalize(self, arr):
      normalize = float(sum(arr[i] for i in arr))
      for i in arr:
         arr[i] = arr[i]/normalize
      return arr
   
   def _influence(self, a, a1):
      (iNew, iTotal) = self.influence
      for i in a:
         a[i] = (a1[i]*iNew+a[i]*(iTotal-iNew)) / iTotal
      return a
      



# hmm = HMM()
# 
# hmm.defHMM(transition_probability, emission_probability, start_probability)
# 
# # print hmm.a
# # print hmm.viterbi(observations2)
# 
# for i in range(0, 5):
#    # print hmm.backward(observations2)
#    hmm.baum_welch(observations2)
#    # print hmm.a
#    for a in hmm.a.itervalues():
#       print sum(a.itervalues())
#    # print hmm.b
#    print "---"
#    for b in hmm.b.itervalues():
#       print sum(b.itervalues())
#    print "\n\n\n"

