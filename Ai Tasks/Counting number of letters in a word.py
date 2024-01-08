#q3
def foo(str):
    t = 0
    size = 0
    for i in str:
        if(i == ' ' 
           or i == ',' 
           or i == '.'):
            
           t = t + 1
        size = size + 1
    # print(cnt) 
    t = t + 1

    # Step 2
    l = 0
    x = 0
    cunt = 0
    legh = [0] * t
    
    for i in str:
        cunt = cunt + 1
        if(i == ' ,' or i == ' ' 
           or i == '.' or cunt == size):
            if(cunt == size):
                l = l + 1
            legh[x] = l
            x = x + 1
            l = 0
        else:
            l = l + 1        
    
    
    count = [0, 0, 0, 0, 0, 0, 0, 0]
    
    for i in legh:
        if(i <= 6):
            count[i] = count[i] + 1
        else:
            count[7] = count[7] + 1
    
    index = 1
    for i in count[1:-1]:
        print(index, " letter words: ", i)
        index = index + 1
    print("More than 6 letter words: ", count[7])

str = "Authenticity is an act of social justice"
foo(str)



