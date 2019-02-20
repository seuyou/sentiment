def iterate_append(test_list):

    try:
        for i, item in enumerate(test_list):
            print(2/item)
    except Exception:
        print("{} by zero".format(i))
        iterate_append(a[i+1:])

a = [1,2,0,3,4]
try:
    for i, item in enumerate(a):
        print(2/item)
except Exception:
    print("{} by zero".format(item))
    iterate_append(a[i+1:])

