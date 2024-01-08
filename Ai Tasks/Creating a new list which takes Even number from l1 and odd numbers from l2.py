#q4
l1=[30,40,22,33,22,11]
l2=[40,30,10,99,53]
result=[]
for i in l1:
    if (i%2 ==0 ):
        result.append(i)
        
for i in l2:
    if(i%2!=0):
        result.append(i)
       
print(result)    