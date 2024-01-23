from libft import ft_flatten_dict
import unittest


class TestPrime(unittest.TestCase):
    def test_ft_flatten_dict(self):
        dict1_test = {'dataset': {'apple': {'healthy': 32}}}
        dict1_test_flatten = {'healthy': 32}
        flattenDict = ft_flatten_dict(dict1_test)
        self.assertDictEqual(flattenDict, dict1_test_flatten)

    def test_ft_flatten_dict2(self):
        dict_test = {'healthy': 32}
        flattenDict = ft_flatten_dict(dict_test)
        self.assertDictEqual(flattenDict, dict_test)

    def test_ft_flatten_dict3(self):
        dict_test = {}
        flattenDict = ft_flatten_dict(dict_test)
        self.assertDictEqual(flattenDict, dict_test)


if __name__ == '__main__':
    unittest.main()
