import random as rd
from matplotlib import pyplot as plt

# The more trials the more the result will be closed to the probabillity! 
def simulate_coin_flips(num_trials):
    # num_trials = 100
    heads = 0
    tails = 0
    p_heads = 0.5

    for i in range(num_trials):
        random_number = rd.random()
        if random_number < p_heads:
            heads = heads + 1
        else:
            tails += 1
    # print("In", num_trials, "trials there were", heads, "heads and", tails, "tails")
    # print("PERCENT HEADS:", 100 * heads/num_trials, "percent")
    percent_heads = heads / num_trials
    return percent_heads

# print(simulate_coin_flips(100))
# print(simulate_coin_flips(1000)
# print(simulate_coin_flips(10000))

def simulate_dice_rolls(N):
    roll_counts = [0,0,0,0,0,0]
    for i in range(N):
        roll = rd.choice([1,2,3,4,5,6])
        index = roll - 1
        roll_counts[index] = roll_counts[index] + 1
    return roll_counts

def show_roll_data(roll_counts):
    number_of_sides_on_die = len(roll_counts)
    for i in range(number_of_sides_on_die):
        number_of_rolls = roll_counts[i]
        number_on_die = i+1
        print(number_on_die, "came up", number_of_rolls, "times")
        
roll_data = simulate_dice_rolls(1000)
show_roll_data(roll_data)

def visualize_one_die(roll_data):
    roll_outcomes = [1,2,3,4,5,6]
    fig, ax = plt.subplots()
    ax.bar(roll_outcomes, roll_data)
    ax.set_xlabel("Value on Die")
    ax.set_ylabel("# rolls")
    ax.set_title("Simulated Counts of Rolls")
    plt.show()
    
roll_data = simulate_dice_rolls(500)
visualize_one_die(roll_data)



#Probabillities of red being red  
p=[0.2, 0.2, 0.2, 0.2, 0.2]
world=['green', 'red', 'red', 'green', 'green']
Z = 'red'
pHit = 0.6
pMiss = 0.2

def sense(p, Z):
    q=[]
    for i in range(len(p)):
        hit = (Z == world[i])
        q.append(p[i] * (hit * pHit + (1-hit) * pMiss))
    s = sum(q)
    for i in range(len(p)):
        q[i]=q[i]/s
    return q
print sense(p,Z)
