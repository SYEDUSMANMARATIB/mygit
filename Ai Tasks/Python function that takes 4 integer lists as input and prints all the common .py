#q4
list1 = [1, 3, 5, 7, 9, 11]
list2 = [5, 8, 14, 17, 18, 19]
list3 = [9, 7, 12, 13, 15]
list4 = [4, 9, 11, 13, 17]

def foo(list1, list2, list3, list4):
    lst = list1 + list2 + list3 + list4
    length = 0
    for i in lst:
        length = length + 1
    # print(length)   
    lst = list2 + list3 + list4
    common = [0] * length
    idx = 0
    
    # Step 2
    for i in list1:
        for j in lst:
            if(i == j):
                common[idx] = i
                idx = idx + 1
    lst = list3 + list4
    for i in list2:
        for j in lst:
            if(i == j):
                common[idx] = i
                idx = idx + 1
    for i in list3:
        for j in list4:
            if(i == j):
                common[idx] = i
                idx = idx + 1
    total = 0
    for i in common:
        if(i == 0):
            break
        total = total + 1
    temp = 0
    idx = 1
    for j in range(total): 
        for i in range(total - 1):
            first = common[i]
            second = common[i + 1]
            if(first > second):
                temp = first
                first = second
                second = temp
            common[i] = first
            common[i + 1] = second
    prev = 0
    for i in range(total):
        if(common[i] != prev):
            print(common[i])
        prev = common[i]


Qfoo(list1, list2, list3, list4)
