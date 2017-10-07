def probability_of_collision(car_1, car_2):
    # car_1 and car_2 will each be strings whose value will either be 
    # "L" for left, "S" for straight, or "R" for right.
    probability = 0.0 # you should change this value based on the directions.
    
    if car_1 == "L":
        # your code here for when car 1 turns left
        if car_2 == "L":
            probability = 0.5
        elif car_2 == "S":
             probability = 0.25
        else: 
             probability = 0.1
    elif car_1 == "S":
        # your code here for when car 1 goes straight
        if car_2 == "L":
            probability = 0.25
        elif car_2 == "S":
             probability = 0.02
        else: 
             probability = 0.1
    else:
        # your code here for when car 1 turns right
        if car_2 == "L":
            probability = 0.1
        elif car_2 == "S":
             probability = 0.1
        else: 
             probability = 0.01
    return probability


# This function is used to test the correctness of your code. You shouldn't
# touch any of the code below here (but feel free to look through it to
# understand what "correct" looks like).
def test():
    num_correct = 0
    
    p1 = probability_of_collision("L", "L"); 
    if p1 == 0.5:
        num_correct += 1
    
    p2 = probability_of_collision("L", "R")
    if p2 == 0.1:
        num_correct += 1
    
    p3 = probability_of_collision("L", "S")
    if p3 == 0.25:
        num_correct += 1
    
    p4 = probability_of_collision("S", "R")
    if p4 == 0.1:
        num_correct += 1
    
    print("You got", num_correct, "out of 4 correct")
    
test()