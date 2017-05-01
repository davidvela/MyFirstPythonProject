from unittest import TestCase

from MyFirstFile import MyFirstClass


class TestMyFirstClass(TestCase):
    def setUp(self):
        self.myClass = MyFirstClass()

    def test_two_roots_1(self):
        self.assertEqual(self.myClass.quadratic_fn(2, 5, 3), (-1.0, -1.5))

    def test_two_roots_2(self):
        self.assertEqual(self.myClass.quadratic_fn(2, -9, 7), (3.5, 1))

    def test_one_root(self):
        # self.assertEqual(self.myClass.quadratic_fn(2, 4, 2), -1)
        self.assertEqual(self.myClass.quadratic_fn(2, 4, 2), (-1.0, -1.0))

    def test_no_roots(self):
        self.assertEqual(self.myClass.quadratic_fn(2, 1, 3), "This equation has no roots")


