import math


class Car:
    def __init__(self, odometer, speed, time):
        self.odometer = odometer
        self.speed = speed
        self.time = time

    def average_speed(self):
        return self.odometer / self.time

    # if __name__ == '__main__':
    # my_car = Car()
    # print("I am a car!")


class MyFirstClass:
    def quadratic_fn(self, a, b, c):  # comments!
        d = b ** 2 - 4 * a * c

        if d >= 0:
            disc = math.sqrt(d)
            root1 = (-b + disc) / (2 * a)
            root2 = (-b - disc) / (2 * a)
            return root1, root2
        else:
            return "This equation has no roots"


if __name__ == '__main__':
    myClass = MyFirstClass()
    while True:
        a = int(input("a: "))
        b = int(input("b: "))
        c = int(input("c: "))
        result = myClass.quadratic_fn(a, b, c)
        print(result)

# MyFirstClass().quadraticFn()
# a 3, b 5 c -4 => works, 0.59 and -2.25
# a 3, b 5 c 6 => error!


# to run: python ./MyFirstFile.py
