class TailCall(Exception):
    def __init__(self, __TCFunc, *args, **kwargs):
        self.func = __TCFunc
        self.args = args
        self.kwargs = kwargs


def TailRec(func):
    if hasattr(func, "__nonTCO"):
        return func

    def funcwrapper(*args, **kwargs):
        # This is so we don't get a stack overflow when
        # calling TailRec funcs
        funcwrapper.__nonTCO = func
        tc = TailCall(func, *args, **kwargs)
        # We can add this to enable debugging of TCO funcs,
        # by keeping track of from whence we came

        # tcs = []
        # tcs.append(tc)
        while True:
            try:
                if hasattr(tc.func, "__nonTCO"):
                    return tc.func.__nonTCO(*tc.args, **tc.kwargs)
                else:
                    return tc.func(*tc.args, **tc.kwargs)
            except TailCall as err:
                tc = err
            #     tcs.append(tc)
            # except Exception as err:
            #     print("Exception in TailCall! Call Stack:")
            #     displayTailCallLog(tcs)
            #     raise err

    return funcwrapper


# def displayTailCallLog(tcs):
#     for tc in tcs:
#         print(tc.func, tc.args,tc.kwargs)

@TailRec
def even(n):
    if n == 0:
        return True
    raise TailCall(odd, n - 1)


@TailRec
def odd(n):
    if n == 0:
        return False
    raise TailCall(even, n - 1)


@TailRec
def fact(n):
    def facthelper(n, i):
        if n == 0:
            return i
        raise TailCall(facthelper, n - 1, i * n)

    raise TailCall(facthelper, n, 1)


def nontcoeven(n):
    if n == 0:
        return True
    return nontcoodd(n - 1)


def nontcoodd(n):
    if n == 0:
        return False
    return nontcoeven(n - 1)


def NonTailRecFact(n):
    def nontailrectfacthelper(n, i):
        if n == 0:
            return i
        return nontailrectfacthelper(n - 1, i * n)

    return nontailrectfacthelper(n, 1)


# TailRec decorator is idempotent, so it doesn't
# break functions that don't raise TailCall
@TailRec
@TailRec
def NotATailRecursiveFunction(a, b):
    return a + b


print(NotATailRecursiveFunction(2, 3))

print(fact(10000))
print(even(1000))
try:
    print("Non TCO Factorial")
    print(NonTailRecFact(1000))
except RuntimeError as err:
    print("RuntimeError: %s" % (format(err),))

try:
    print("Non TCO Even function")
    print(nontcoeven(1000))
except RuntimeError as err:
    print("RuntimeError: %s" % (format(err),))