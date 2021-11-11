def star(func):
    def inner(*args, **kwargs):
        print("*",args,kwargs)
        func(*args, **kwargs)
        print("*" * 30)
    return inner


def percent(func):
    def inner(*args, **kwargs):
        print("%",args,kwargs)
        func(*args, **kwargs)
        print("%" * 30)
    return inner

def r(func):
    def inner(*args, **kwargs):
        print("r",args,kwargs)
        args = (*args,"hii")
        func(*args, **kwargs)
        print("r" * 30)
    return inner

@star
@percent
@r
def printer(msg,*args, **kwargs):
    print(msg,args, kwargs)


printer("Hello")