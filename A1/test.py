
# from collections import ChainMap
# params1 = { 'a': 1, 'b': 2, 'c': 3 }
# params2 = { 'd': 1, 'e': 2, 'f': 3 }
# params5 = { 'i': 1, 'h': 2, 'g': 3 }
# params3 = ChainMap(params1, params2)
# params4 = ChainMap(params3, params5)
# # params3['a'] = 10
# # print(params1['a'])
# # print(params3['a'])
# # params4['d'] = 10
# # params4['g'] = 11
# # breakpoint()
# # params1['a'] = -5
# # breakpoint()
# # print('lol')

# class Test:
#     def __init__(self, params) -> None:
#         self.params = params
#         pass
#     def modify_params(self, params):
#         params['a'] = 1000


# t = Test(params4)
# t.params['a'] = 100
# params4['a'] = 1000
# print(params4['a'])
# print(t.params['a'])
import numpy as np
params = {}
# a  = np.array([1,2,3]) 
# b = a
# params[1] = [a,b]
# a +=1
# print(params)
# print(b)
# print(a)
#----------
# b = [10,11,12]
# a = b
# a += [1]
# a *=2 # returns references!!
# print(a)
# print(b)
#------------------
# def increment(x):
#     print(id(x))
#     x = x + 1
#     print(id(x))
#     return x

# y = 3
# print(id(y))
# y = increment(y)
# print(id(y))
# print(y)

#------------------

d = {'a': 1, 'b': 2, 'c': 3}
for key, value in d.items():
    value += 1

print(d)


a = [1,2,3,4,5]
for k in a:
    k +=1
print(a)