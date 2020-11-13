import ctypes as ct

yourString = "somestring"
yourDLL = ct.CDLL("/home/mahbub/ImageProcessing/test.cpp") # assign the dll to a variable
cppFunc = yourDLL.changeString # assign the cpp func to a variable
cppFunc.restype = ct.c_char_p # set the return type to a string
returnedString = cppfunc(yourString.encode('ascii')).decode()

print(returnedString)