#q2
def strRev(sentence):
    print('Enter thr sentence which you what to reverse:')
    sentence = input()
    return ' '.join(word[::-1] for word in sentence.split(" "))
print(strRev(sentence))