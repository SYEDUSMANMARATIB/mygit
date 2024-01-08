#q6
string=input('Enter the String:')
leter=0
num=0
special=0
for i in string:
    if(i.isdigit()):
        num=num+1
    elif(i.isalpha()):
        leter=leter+1
    else:   
        special=special+1 
print('Letters in a Sentance:')    
print(leter)
print('numbers in a Sentance:')    
print(num)
print('unkowns in a Sentance:') 
print(special)