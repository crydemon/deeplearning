# 中缀转前缀 参考算法
# 1）求输入串的逆序。
# 2）检查输入的下一元素。
# 3）假如是操作数，把它添加到输出串中。
# 4）假如是闭括号，将它压栈。
# 5）假如是运算符，则
# i)假如栈空，此运算符入栈。
# ii)假如栈顶是闭括号，此运算符入栈。
# iii)假如它的优先级高于或等于栈顶运算符，此运算符入栈。
# iv)否则，栈顶运算符出栈并添加到输出串中，重复步骤5。
# 6）假如是开括号，栈中运算符逐个出栈并输出，直到遇到闭括号。闭括号出栈并丢弃。
# 7）假如输入还未完毕，跳转到步骤2。
# 8）假如输入完毕，栈中剩余的所有操作符出栈并加到输出串中。
# 9）求输出串的逆序。
def opOrder(op1, op2):
    order_dic = {'*': 4, '/': 4, '+': 3, '-': 3}
    if op1 == '(' or op2 == '(':
        return False
    elif op2 == ')':
        return True
    else:
        if order_dic[op1] < order_dic[op2]:
            return False
        else:
            return True


def infix2prefix(string):
    prefix = ''
    stack = []
    string_tmp = []
    for s in string[::-1]:
        if s == '(':
            string_tmp += ')'
        elif s == ')':
            string_tmp += '('
        else:
            string_tmp += s
    for s in string_tmp:
        # 检测字符串是否只由字母组成。
        if s.isalpha():
            prefix = s + prefix
        else:
            while len(stack) and opOrder(stack[-1], s):
                op = stack.pop()
                prefix = op + prefix
            if len(stack) == 0 or s != ')':
                stack.append(s)
            else:
                stack.pop()
    if len(stack):
        prefix = ''.join(stack) + prefix
    return prefix


if __name__ == '__main__':
    for string in ['A+B*C', '(A+B)*C']:
        print(string, '==>', infix2prefix(string))
