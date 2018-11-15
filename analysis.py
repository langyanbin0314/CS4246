import numpy as np
import matplotlib.pyplot as plt
def getStats(filename):

    x = np.loadtxt(filename)

    hundred_episodes = []

    for i in range(len(x)-100):
        a = x[i:i+100]
        avg = np.average(a)
        hundred_episodes.append(avg)

    print(filename, np.max(hundred_episodes))
    return hundred_episodes


# a= getStats('noDuel.txt')
# b= getStats('duelMax.txt')
# c= getStats('duelAvg.txt')
#
# plt.plot(np.arange(len(a)), a, 'r', label = 'noDuel')
# plt.plot(np.arange(len(b)), b, 'b', label = 'duelMax')
# plt.plot(np.arange(len(c)), c, 'g', label = 'duelAvg')
# plt.xlabel('the nth 100-episode')
# plt.ylabel('100-episode average reward')
# plt.legend()
# plt.show()


# a= getStats('e-greedy.txt')
# b= getStats('greedy.txt')
# c= getStats('boltzmann.txt')
# d= getStats('e-time-greedy.txt')
#
# plt.plot(np.arange(len(b)), b, 'b', label = 'greedy')
# plt.plot(np.arange(len(a)), a, 'r', label = 'e-greedy linear Annealing')
# plt.plot(np.arange(len(d)), d, 'y', label = 'e-greedy 1/t')
# plt.plot(np.arange(len(c)), c, 'g', label = 'boltzmann')
#
# plt.xlabel('the nth 100-episode')
# plt.ylabel('100-episode average reward')
# plt.legend()
# plt.show()

a= np.loadtxt('e-greedy.txt')
b= np.loadtxt('greedy.txt')
c= np.loadtxt('boltzmann.txt')
d= np.loadtxt('e-time-greedy.txt')

plt.plot(np.arange(len(b)), b, 'b', label = 'greedy')
plt.plot(np.arange(len(a)), a, 'r', label = 'e-greedy linear Annealing')
plt.plot(np.arange(len(d)), d, 'y', label = 'e-greedy 1/t')
plt.plot(np.arange(len(c)), c, 'g', label = 'boltzmann')

plt.xlabel('episode')
plt.ylabel('reward')
plt.legend()
plt.show()


# a= np.loadtxt('e-greedy.txt')
# b= np.loadtxt('greedy.txt')
# c= np.loadtxt('boltzmann.txt')
#
# plt.plot(np.arange(len(a)), a, 'r', label = 'e-greedy')
# plt.plot(np.arange(len(b)), b, 'b', label = 'greedy')
# plt.plot(np.arange(len(c)), c, 'g', label = 'boltzmann')
# plt.xlabel('episode')
# plt.ylabel('reward')
# plt.legend()
# plt.show()

# a= getStats('ac.txt')
# b= getStats('ps_3000.txt')
#
# plt.plot(np.arange(len(a)), a, 'r', label = 'actor-critic')
# plt.plot(np.arange(len(b)), b, 'b', label = 'policy search')
# plt.xlabel('the nth 100-episode')
# plt.ylabel('100-episode average reward')
# plt.legend()
# plt.show()


# a= np.loadtxt('ac.txt')
# b= np.loadtxt('ps.txt')
#
# plt.plot(np.arange(len(a)), a, 'r', label = 'actor-critic')
# plt.plot(np.arange(len(b)), b, 'b', label = 'policy search')
# plt.xlabel('the nth 100-episode')
# plt.ylabel('100-episode average reward')
# plt.legend()
# plt.show()