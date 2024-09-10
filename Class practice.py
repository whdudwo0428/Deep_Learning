class Person:
    def __init__(self,name, age):
        self.name = name
        self.age = age

    def f_print(self):
        print("My is name{0}".format(self.name))
        print("My is age{0}".format(self.age))

#객체 생성하기
p1 = Person("홍길동",20)
p2 = Person("홍길동",30)

print(p1)