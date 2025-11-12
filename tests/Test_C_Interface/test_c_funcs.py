import ctypes 
import numpy as np
import pylab as py




c_test = ctypes.cdll.LoadLibrary("c_test.so")



def c_function0():
    """
    This is test function to test printing from a called shared object library
    in C

    Returns
    -------
    None.

    """
    c_test.Func0()


def c_function1():
    """
    This is test function to confirm int, float, and double datatypes in C size

    Returns
    -------
    None.

    """
    c_test.Func1()


def c_function2(int_var, float_var, double_var):
    """
    This is a test function to pass ints, floats, and doubles to a C shared
    object library. The values will be copied and sent to C. Any modifcations
    made in the shared object will not be returned.

    Parameters
    ----------
    int_var : python int
        Int variable to be passed to a C shared object as an int
    float_var : python float
        float variable to be passed to a C shared object as a float
    double_var : python float
        float variable to be passed to a C shared object as a double

    Returns
    -------
    None.

    """
    #Hard Code - following will pass (python ints can be passed)    
    c_test.Func2(ctypes.c_int(1), ctypes.c_float(2.0), ctypes.c_double(3.0))
    c_test.Func2(ctypes.c_int(1), ctypes.c_float(2), ctypes.c_double(3))
    c_test.Func2(1, ctypes.c_float(2), ctypes.c_double(3.0))

    #Hard Code - This test will fail 
    #c_test.Func2(1, 2, 3)

    #Passing python object paramters
    c_test.Func2(ctypes.c_int(int_var),ctypes.c_float(float_var),ctypes.c_double(double_var))
    #c_test.Func2(int_var,ctypes.c_float(float_var))




def c_function3(int_var, float_var, double_var):
    """
    This is a test function to pass pointers to ints, floats, and doubles to a
    C shared object library. The values will be copied and sent to C. Any
    modifcations made in the shared object will not be returned.

    Parameters
    ----------
    int_var : python int
        Int variable to be passed to a C shared object as an int
    float_var : python float
        float variable to be passed to a C shared object as a float
    double_var : python float
        float variable to be passed to a C shared object as a double

    Returns
    -------
    c_type int,float, and double by reference.

    """
    #Hard Code - following will pass (python ints can be passed)        
    iv_hc = ctypes.c_int(1)
    fv_hc = ctypes.c_float(2.0)
    dv_hc = ctypes.c_double(3.0)
    iv_hcp = ctypes.pointer(iv_hc)
    fv_hcp = ctypes.pointer(fv_hc)
    dv_hcp = ctypes.pointer(dv_hc)
    c_test.Func3(iv_hcp,fv_hcp,dv_hcp)
    print("This is the returned int pointer:",iv_hc.value)
    print("This is the returned float pointer:",fv_hc.value)
    print("This is the returned double pointer:",dv_hc.value,"\n")

    #Passing python object paramters by pointers      
    iv_hc = ctypes.c_int(int_var)
    fv_hc = ctypes.c_float(float_var)
    dv_hc = ctypes.c_double(double_var)
    iv_hcp = ctypes.pointer(iv_hc)
    fv_hcp = ctypes.pointer(fv_hc)
    dv_hcp = ctypes.pointer(dv_hc)
    c_test.Func3(iv_hcp,fv_hcp,dv_hcp)
    print("This is the python int pointer:",int_var)
    print("This is the python float pointer:",float_var)
    print("This is the python double pointer:",double_var,"\n")
    print("This is the returned int pointer:",iv_hc.value)
    print("This is the returned float pointer:",fv_hc.value)
    print("This is the returned double pointer:",dv_hc.value,"\n")

    #Passing python object paramters by byref 
    iv_hc = ctypes.c_int(int_var)
    fv_hc = ctypes.c_float(float_var)
    dv_hc = ctypes.c_double(double_var)
    c_test.Func3(ctypes.byref(iv_hc),\
                 ctypes.byref(fv_hc),\
                 ctypes.byref(dv_hc))
    print("This is the python int pointer:",int_var)
    print("This is the python float pointer:",float_var)
    print("This is the python double pointer:",double_var,"\n")
    print("This is the returned int pointer:",iv_hc.value)
    print("This is the returned float pointer:",fv_hc.value)
    print("This is the returned double pointer:",dv_hc.value,"\n")



def c_function4(arr_size):
    """
    This is a test function to pass numpy arrays to a C shared object library.
    The values will not be copied and pointers will be sent to C. Any
    modifcations made in the shared object will be returned.

    Parameters
    ----------
    arr_size : (arr_size) np.array
        A one dimensional numpy array

    Returns
    -------
    The arrays by reference.

    """
    
    #First version using ctypes
    i_arr1 =  np.arange(arr_size, dtype=np.int32)
    i_arr_c1 = ctypes.c_void_p(i_arr1.ctypes.data)    
    c_test.Func4i(ctypes.c_int(arr_size),i_arr_c1)
    print("This is the returned int array:\n",i_arr1)
    i_arr1[1] = 20
    print("This is the python modified int array:\n",i_arr1,"\n\n")

    #Second version using numpy's ctypeslib
    i_arr2 =  np.arange(arr_size, dtype=np.int32)
    i_arr_c2 = np.ctypeslib.as_ctypes(i_arr2)
    c_test.Func4i(ctypes.c_int(arr_size),i_arr_c2)
    print("This is the returned int array:\n",i_arr2)
    i_arr2[1] = 20
    print("This is the python modified int array:\n",i_arr2,"\n\n")
    
    #3rd version using a creating pointesrs
    i_arr3 =  np.arange(arr_size, dtype=np.int32)
    c_int_p = ctypes.POINTER(ctypes.c_int)
    i_arr_c3 = i_arr3.ctypes.data_as(c_int_p)    
    c_test.Func4i(ctypes.c_int(arr_size),i_arr_c3)
    print("This is the returned int array:\n",i_arr3)
    i_arr3[1] = 20
    print("This is the python modified int array:\n",i_arr3,"\n\n")


    #3rd version using a creating pointesrs
    i_arr4 =  np.arange(arr_size, dtype=np.int32)
    f_arr4 =  np.arange(arr_size, dtype=np.float32)
    d_arr4 =  np.arange(arr_size, dtype=np.float64)
    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_double_p = ctypes.POINTER(ctypes.c_double)
    i_arr_c4 = i_arr4.ctypes.data_as(c_int_p)    
    f_arr_c4 = f_arr4.ctypes.data_as(c_float_p)
    d_arr_c4 = d_arr4.ctypes.data_as(c_double_p)
    c_test.Func4i(ctypes.c_int(arr_size),i_arr_c4)
    c_test.Func4f(ctypes.c_int(arr_size),f_arr_c4)
    c_test.Func4d(ctypes.c_int(arr_size),d_arr_c4)
    print("This is the returned int array:\n",i_arr4)
    i_arr4[1] = 20
    print("This is the python modified int array:\n",i_arr4)
    print("This is the returned float array:\n",f_arr4)
    f_arr4[1] = 20
    print("This is the python modified float array:\n",f_arr4)
    print("This is the returned double array:\n",d_arr4)
    d_arr4[1] = 20
    print("This is the python modified double array:\n",d_arr4,"\n\n")


def c_function5(nAx0, nAx1):
    """
    This is a test function to pass 2d numpy array to a C shared object library
    and confirm row major.

    Parameters
    ----------
    nAx0 : int
        A size of frist axis of the 2 dimensional numpy array

    nAx1 : int
        A size of second axis of the 2 dimensional numpy array

    Returns
    -------
    None

    """
    
    f2_array =  np.arange(nAx0*nAx1, dtype=np.float32).reshape([nAx0,nAx1])
    c_test.Func5(ctypes.c_int(nAx0),ctypes.c_int(nAx1),ctypes.c_void_p(f2_array.ctypes.data))
    print("Numpy shape: ",f2_array.shape,"\n\n")


class Data(ctypes.Structure):
     _fields_ = [("n", ctypes.c_int),
                ("in_arr_f", ctypes.POINTER(ctypes.c_double)),
                ("out_arr_f", ctypes.POINTER(ctypes.c_double)),
                ("in_arr_d", ctypes.POINTER(ctypes.c_double)),
                ("out_arr_d", ctypes.POINTER(ctypes.c_double))]

def c_function6():
    n=5
    in_arr_f = np.linspace(1.0,n,n)
    out_arr_f = np.zeros(n,float)
    
    data = Data(ctypes.c_int(n),
                np.ctypeslib.as_ctypes(in_arr_f),
                np.ctypeslib.as_ctypes(out_arr_f))
    
    c_test.Func6(ctypes.byref(data))
    
    print("This is the returned double array:\n",in_arr_f)
    print("This is the returned double array:\n",out_arr_f,"\n")
    
    in_arr_f[0] = 10.0
    in_arr_f[1] = 20.0
    in_arr_f[2] = 30.0
    in_arr_f[3] = 40.0
    in_arr_f[4] = 50.0
    
    c_test.Func6(ctypes.byref(data))
    print("This is the returned double array:\n",in_arr_f)
    print("This is the returned double array:\n",out_arr_f,"\n")
    


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


import psutil

def c_function_mem():
    N = 1024*1024*128*4

    process = psutil.Process()
    m1 = process.memory_info().rss/1024/1024/1024

    f_arr3 =  np.arange(N, dtype=np.float32)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    f_arr_c3 = f_arr3.ctypes.data_as(c_float_p)
    
    process = psutil.Process()
    m2 = process.memory_info().rss/1024/1024/1024
    print(m2-m1)
    
    f_arr2 =  np.arange(N, dtype=np.float32)
    f_arr_c2 = np.ctypeslib.as_ctypes(f_arr2)
    process = psutil.Process()
    m3 = process.memory_info().rss/1024/1024/1024
    print(m3-m2)

    f_arr1 =  np.arange(N, dtype=np.float32)
    f_arr_c1 = ctypes.c_void_p(f_arr1.ctypes.data)
    process = psutil.Process()
    m4 = process.memory_info().rss/1024/1024/1024
    print(m4-m3)


#Tests library call and printing
c_function0()


#Tests C variable prints
c_function1()


#Tests passing variables
c_function2(1, 2.0, 3.0)


#Tests passing int arrays
c_function3(1,2.0, 3.0)


#Tests passing int arrays
c_function4(5)


#Tests passing 2d arrays
c_function5(3,4)


#Tests reference assignments in float arrays
c_function6()


#Tests structures
#c_function7()


#Tests structures
#c_function8()

#Tests return values
#c_function10(1,1.0)

