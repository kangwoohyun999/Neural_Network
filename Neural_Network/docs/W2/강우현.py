import math

class Vector2D:
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def magnitude(self):
        """벡터의 크기 계산"""
        return math.sqrt(self.x**2 + self.y**2)
    
    def add(self, other):
        """벡터 덧셈"""
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def substract(self,other):
        """벡터 뺄셈"""
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def dot(self,other):
        """벡터 내적"""
        return self.x*other.x + self.y*other.y
    
    def __str__(self):
        return f"Vector({self.x},{self.y})"
    
    
v1=Vector2D(3,4)
v2 = Vector2D(1,2)

v4 = v1.substract(v2)
print(v4)

print(v1.dot(v2))
