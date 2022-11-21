

class Kek():
    def __init__(self):
        self.value = 10

    
    def __add__(self, other):
        print(self)
        return self.value + other.value
    
    def __radd__(self, other):
        print(self)
        return self.value + other
    
    def __repr__(self) -> str:
        return 202
    
    def __str__(self) -> str:
        return "kek"

a = Kek()
b = Kek()
print(a + b)