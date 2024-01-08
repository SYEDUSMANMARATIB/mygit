#q5

d1 = {'k1': 1, 'k2': 3, 'k3': 2, 'k4':3}
d2 = {'k1': 1, 'k2': 2, 'k3': 2}

def foo(d1, d2):
    zu = {}
    lee = d1.keys()
    
    for x in lee:
        #comparison only
        if(d1.get(x) == d2.get(x)):
            print(x, ":", d1.get(x), " ")
    print("are present in both the dictionaries.")

foo(d1, d2)