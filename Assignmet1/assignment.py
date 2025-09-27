
import random

################## LIST ##############################

#Q1. Create a list of 5 random numbers and print the list.
ran_list = random.sample(range(1,100),5)
print(ran_list)
# [40, 31, 16, 56, 51]

#Q2. Insert 3 new values to the list and print the updated list.

ran_list.append(1)
ran_list.append(2)
ran_list.append(3)

print(ran_list)
# [40, 31, 16, 56, 51, 1, 2, 3]

#Q3. Try to use a for loop to print each element in the list.

for i in range(len(ran_list)):
    print(f"""list {i+1} item""",ran_list[i])

# list 1 item 40
# list 2 item 31
# list 3 item 16
# list 4 item 56
# list 5 item 51
# list 6 item 1
# list 7 item 2
# list 8 item 3


################### Dictionary ####################


#Q1. Create a dictionary with keys 'name', 'age', and 'address' and values 'John', 25, and 'New York' respectively.

dict_person = {"name":"John","age":25, "address":"New York"}
print(dict_person)
# {'name': 'John', 'age': 25, 'address': 'New York'}

#Q2. Add a new key-value pair to the dictionary created in Q1 with key 'phone' and value ''.

dict_person["phone"] = "1234567890"
print(dict_person)
# {'name': 'John', 'age': 25, 'address': 'New York', 'phone': '1234567890'}

#######################Topic: Set ###############################

#Q1.Create a set with values 1, 2, 3, 4, and 5.

set_num = set(range(1,6))
print(set_num)
# {1, 2, 3, 4, 5}

#Q2. Add the value 6 to the set created in Q1.

set_num.add(6)
print(set_num)
#{1, 2, 3, 4, 5, 6}

#Q3. Remove the value 3 from the set created in Q1.

set_num.remove(3)
print(set_num)
# {1, 2, 4, 5, 6}

####################Topic:Tuple######################################

#Q1. Create a tuple with values 1, 2, 3, and 4

tuple_num = tuple(range(1,5))
print(tuple_num)
# (1, 2, 3, 4)

#Q2. Print the length of the tuple created in Q1.
print(len(tuple_num))
# 4