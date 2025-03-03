if __name__ == '__main__':
    print('Hello, world!')

    # two object arguments
    print('Hello,', 'world!')
    # sep argument
    print('Hello', 'world', sep='-')
    # end argument
    print('Hello', 'world', end=' ')
    # exclamation mark
    print('!', flush=True)

    # common built-in functions
    # Return the absolute value of a number
    print(abs(-5), end=' ')
    # Return a number rounded to a specified number of decimal places.
    print(round(3.14159, 2), end=' ')
    # Return the largest item in an iterable or the largest.
    print(max(1, 3, 2), end=' ')
    # Sums the items of an iterable and returns the total.
    print(sum([1, 2, 3]), end=' ') 
    # Return an integer object constructed from a string.
    print(int('42'), end=' ') 
    # Return a floating point number constructed from a string.
    print(float('3.14'), end= ' ')
    # str is the built-in string class.
    print(str(42), end=' ')
    # Return the length (the number of items) of an object.
    print(len('hello'))
    # A mutable sequence type.
    print(list('hello'))
    # Create a new dictionary.
    print(dict(a=1, b=2))
    # Return a new sorted list from the items in iterable.
    print(sorted([3, 1, 2, 7, 9, 8]))
    # Return a reverse iterator.
    print(list(reversed([1, 2, 3, 4, 5, 6, 7])))
    # With one argument, return the type of object.
    print(type(42)) 
    # Return True if the object argument is an instance.
    print('42 is instance of int:', isinstance(42, int))
