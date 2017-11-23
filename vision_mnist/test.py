import pickle


try:
    f = open('test.txt', 'rb')
    li = pickle.load(f)
    f.close()
except:
    pass
print(li)
print(li[0])

