import ctypes 
import numpy as np
import pylab as py




c_test = ctypes.cdll.LoadLibrary("c_test.so")



def c_function0():
    c_test.Func0()


def c_function1():
    c_test.Func1()


def c_function2(int_var, float_var):
    
    #Hard Code
    c_test.Func2(ctypes.c_int(12), ctypes.c_float(24.0))
    c_test.Func2(ctypes.c_int(12), ctypes.c_float(24))
    c_test.Func2(12, ctypes.c_float(24.0))

    c_test.Func2(ctypes.c_int(int_var),ctypes.c_float(float_var))
    c_test.Func2(int_var,ctypes.c_float(float_var))


def c_function3(arr_size):
    i_arr1 =  np.arange(arr_size, dtype=np.int32)
    i_arr_c1 = ctypes.c_void_p(i_arr1.ctypes.data)
    c_test.Func3(ctypes.c_int(arr_size),i_arr_c1)

    i_arr2 =  np.arange(arr_size, dtype=np.int32)
    i_arr_c2 = np.ctypeslib.as_ctypes(i_arr2)
    c_test.Func3(ctypes.c_int(arr_size),i_arr_c2)

    i_arr3 =  np.arange(arr_size, dtype=np.int32)
    c_int_p = ctypes.POINTER(ctypes.c_int)
    i_arr_c3 = i_arr3.ctypes.data_as(c_int_p)
    c_test.Func3(ctypes.c_int(arr_size),i_arr_c3)


def c_function4(arr_size):
    #Tests passing float arrays
    f_arr1 =  np.arange(arr_size, dtype=np.float32)
    f_arr_c1 = ctypes.c_void_p(f_arr1.ctypes.data)
    c_test.Func4(ctypes.c_int(arr_size),f_arr_c1)

    f_arr2 =  np.arange(arr_size, dtype=np.float32)
    f_arr_c2 = np.ctypeslib.as_ctypes(f_arr2)
    c_test.Func4(ctypes.c_int(arr_size),f_arr_c2)

    f_arr3 =  np.arange(arr_size, dtype=np.float32)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    f_arr_c3 = f_arr3.ctypes.data_as(c_float_p)
    c_test.Func4(ctypes.c_int(arr_size),f_arr_c3)


def c_function5(nAx0, nAx1):
    f2_array =  np.arange(nAx0*nAx1, dtype=np.float32).reshape([nAx0,nAx1])
    c_test.Func5(ctypes.c_int(nAx0),ctypes.c_int(nAx1),ctypes.c_void_p(f2_array.ctypes.data))


def c_function6():
    f_arr1 =  np.arange(10, dtype=np.float32)
    f_arr_c1 = ctypes.c_void_p(f_arr1.ctypes.data)
    print(f_arr1)
    c_test.Func6(ctypes.c_int(2), ctypes.c_int(10),f_arr_c1)
    print(f_arr1)
    f_arr1[6] = 33
    print(f_arr1)
    c_test.Func6(ctypes.c_int(8), ctypes.c_int(10),f_arr_c1)
    print(f_arr1)
    
    print("\n\n")
    f_arr2 =  np.arange(10, dtype=np.float32)
    f_arr_c2 = np.ctypeslib.as_ctypes(f_arr2)
    print(f_arr2)
    c_test.Func6(ctypes.c_int(2), ctypes.c_int(10),f_arr_c2)
    print(f_arr2)
    f_arr2[6] = 33
    print(f_arr2)
    c_test.Func6(ctypes.c_int(8), ctypes.c_int(10),f_arr_c2)
    print(f_arr2)
    
    print("\n\n")
    f_arr3 =  np.arange(10, dtype=np.float32)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    f_arr_c3 = f_arr3.ctypes.data_as(c_float_p)
    print(f_arr3)
    c_test.Func6(ctypes.c_int(2), ctypes.c_int(10),f_arr_c3)
    print(f_arr3)
    f_arr3[6] = 33
    print(f_arr3)
    c_test.Func6(ctypes.c_int(8), ctypes.c_int(10),f_arr_c3)
    print(f_arr3)


class Data(ctypes.Structure):
     _fields_ = [("n", ctypes.c_int),
                ("ina", ctypes.POINTER(ctypes.c_double)),
                ("outa", ctypes.POINTER(ctypes.c_double))]

def c_function7():
    n=5
    outa = np.zeros(n,float)
    ina = np.linspace(1.0,n,n)
    
    data = Data(ctypes.c_int(n),
                np.ctypeslib.as_ctypes(ina),
                np.ctypeslib.as_ctypes(outa))
    
    print("initial array",ina)
    print("final array",outa)
    c_test.Func7(ctypes.byref(data))
    
    print("initial array",ina)
    print("final array",outa)
    
    ina[2] = 10.0
    
    
    print("initial array",ina)
    print("final array",outa)
    c_test.Func7(ctypes.byref(data))
    
    print("initial array",ina)
    print("final array",outa)


class Ray(ctypes.Structure):
     _fields_ = [("n", ctypes.c_int),
                ("X", ctypes.POINTER(ctypes.c_int)),
                ("Y", ctypes.POINTER(ctypes.c_int)),
                ("Z", ctypes.POINTER(ctypes.c_int)),
                ("L", ctypes.POINTER(ctypes.c_float))]

def c_function8():
    n=10
    X = np.arange(n, dtype=np.int32)
    Y = np.arange(n, dtype=np.int32)*2
    Z = np.arange(n, dtype=np.int32)*3
    L = np.linspace(1,n,n, dtype=np.float32)
    
    ray1 = Ray(ctypes.c_int(n),
                np.ctypeslib.as_ctypes(X),
                np.ctypeslib.as_ctypes(Y),
                np.ctypeslib.as_ctypes(Z),
                np.ctypeslib.as_ctypes(L))
    
    c_test.Func8(ctypes.byref(ray1))
    
    
    rays = (Ray*2)()
    n = 4
    X = np.arange(n, dtype=np.int32)
    Y = np.arange(n, dtype=np.int32)*2
    Z = np.arange(n, dtype=np.int32)*3
    L = np.linspace(1,n,n, dtype=np.float32)
    rays[0].n = n
    rays[0].X = np.ctypeslib.as_ctypes(X)
    rays[0].Y = np.ctypeslib.as_ctypes(Y)
    rays[0].Z = np.ctypeslib.as_ctypes(Z)
    rays[0].L = np.ctypeslib.as_ctypes(L)
    
    n = 5
    X = np.arange(n, dtype=np.int32)
    Y = np.arange(n, dtype=np.int32)*2
    Z = np.arange(n, dtype=np.int32)*3
    L = np.linspace(1,n,n, dtype=np.float32)
    rays[1].n = n
    rays[1].X = np.ctypeslib.as_ctypes(X)
    rays[1].Y = np.ctypeslib.as_ctypes(Y)
    rays[1].Z = np.ctypeslib.as_ctypes(Z)
    rays[1].L = np.ctypeslib.as_ctypes(L)
    
    c_test.Func8(ctypes.byref(rays[0]))
    c_test.Func8(ctypes.byref(rays[1]))

    c_test.Func9(ctypes.byref(rays), ctypes.c_int(2))



def c_function10(int_var, float_var):
    #Tests passing float arrays
    
    i = ctypes.c_int(int_var)
    i_c = ctypes.pointer(i)
    c_test.Func10(i_c, ctypes.c_float(float_var))
    print(int_var, i, i_c, i.value)

#Tests library call and printing
#c_function0()


#Tests C variable prints
#c_function1()


#Tests passing variables
#c_function2(2, 3.0)


#Tests passing int arrays
#c_function3(5)


#Tests passing int arrays
#c_function4(8)


#Tests passing 2d arrays
#c_function5(3,4)


#Tests reference assignments in float arrays
#c_function6()


#Tests structures
#c_function7()


#Tests structures
#c_function8()

#Tests return values
c_function10(1,1.0)

