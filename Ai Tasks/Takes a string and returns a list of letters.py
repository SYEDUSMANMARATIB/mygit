#q1
def strLet(word,flag):
   
    if flag==0:
        count =0
        for i in list(word):
            if count%2 !=0:
                print(i)
                count = count+1
            else:
                count = count+1
    else:
        count=1
        for i in list(word):
            if count%2 != 0:
                print(i)
                count =count+1
            else:
                count=count+1
                
            
print(strLet('Assignment', 0))
print(strLet('Assignment',1 ))  