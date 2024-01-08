#q5
def remDup(l1):
    ans=[]
    for i in l1:
        if i not in ans:
            ans.append(i)
    print(ans)
remDup([30,40,22,33,22,11])