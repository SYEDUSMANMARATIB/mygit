#8
def count(str):
    i=str.split()
    j=dict()
    for word in i:
        if word in j:
            j[word] +=1
        else:
            j[word] =1
            
   
    for i in sorted (j): 
         print(i, ": ", j[i] ,end = "\n")
   # return j
count( 'I love python but I am confused whether to select python 2 or python3')
          