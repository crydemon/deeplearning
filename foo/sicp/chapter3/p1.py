def make_account(balance):
    def withdraw(amount):
        return make_account(balance - amount)

    def deposit(amount):
        return make_account(balance + amount)

    def dispatch(m):
        if m == 'withdraw':
            return withdraw
        elif m == 'deposit':
            return deposit
        elif m == 'check':
            return balance
        else:
            raise Exception("Unknown request")

    return dispatch


def acc():
    return make_account(100)


def make_accmulator(a):
    return lambda x: a + x


def A():
    return make_accmulator


if __name__ == '__main__':
    print(A()(5)(10))
    # print(acc()('withdraw')(50)('check'))
