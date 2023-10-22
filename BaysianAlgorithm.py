age = ['youth','youth','youth','middle','senior','senior','middle','youth','middle','senoir']
income = ['low','high','high','medium','low','low','high','medium','low','medium']
student = ['no','yes','no','yes','yes','yes','yes','no','no','yes']
credit_rating = ['fair','excellent','fair','fair','fair','excellent','excellent','fair','excellent','fair']
buys_computer = ['yes','no','yes','yes','yes','yes','yes','no','yes','yes',]

p_y = buys_computer.count('yes')/10 
p_n = buys_computer.count('no')/10 
p_yes = buys_computer.count('yes') 
p_no = buys_computer.count('no') 

p_r_yes = 0
p_r_no = 0

for i in range(10):
    if age[i] == 'youth' and buys_computer[i] == 'yes':
        p_r_yes += 1
    elif age[i] == 'youth' and buys_computer[i] == 'no':
        p_r_no += 1
        p_s_yes = 0 
        p_s_no = 0 

for i in range(10):
    if income[i] == 'medium' and buys_computer[i] == 'yes':
        p_s_yes += 1
    elif income[i] == 'medium' and buys_computer[i] == 'no':
        p_s_no += 1
        p_d_yes = 0
        p_d_no = 0

for i in range(10):
    if student[i] == 'yes' and buys_computer[i] == 'yes':
        p_d_yes += 1
    elif student[i] == 'yes' and buys_computer[i] == 'no':
        p_d_no += 1
        p_x_yes = 0
        p_x_no = 0

for i in range(10):
    if credit_rating[i] == 'fair' and buys_computer[i] == 'yes':
        p_x_yes += 1
    elif credit_rating[i] == 'fair' and buys_computer[i] == 'no':
        p_x_no += 1

prob_yes =  (p_r_yes/p_yes) * (p_d_yes/p_yes) * (p_s_yes/p_yes) * (p_x_yes/p_yes) * p_y
prob_no = (p_r_no/p_no) * (p_d_no/p_no) * (p_s_no/p_no) * (p_s_no/p_no) * p_n 

print(prob_yes , prob_no) 

if prob_yes > prob_no: 
    print("Hence it is classified as Yes") 
else: 
    print("Hence it is classified as No")

