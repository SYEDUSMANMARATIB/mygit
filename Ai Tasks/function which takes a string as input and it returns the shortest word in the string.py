
def words(str1):
    word = ""
    words1 = []
    str1 = str1 + " "
    for i in range(0, len(str1)):
        if (str1[i] != ' '):
            word = word + str1[i]
        else:
            words1.append(word)
            word = ""

    small =  words1[0]
 # Find smallest
    for k in range(0, len(words1)):
        if (len(small) > len(words1[k])):
            small = words1[k]
    return small
str1 = "This is a very long sentence boy."
print("STRING IS:", str1)
small = words(str1)
print("Smallest word: " + small)
