class CStack:
    def __init__(self, k = list()):
        self.Stack = k
    def empty(self):
        return len(self.Stack) == 0
    def peek(self):
        return self.Stack[-1]
    def push(self, k):
        self.Stack.append(k)
        return True
    def pop(self):
        if len(self.Stack) != 0:
            return self.Stack.pop()
        return False
    