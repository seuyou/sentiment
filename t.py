def iterate_append(test_list):

    try:
        for i, item in enumerate(test_list):
<<<<<<< HEAD
            b.append(item)
    except Exception:
        b.append(0)
        iterate_append(a[i+1:])

a = [1,2,0,3,4]
b = []
try:
    for i, item in enumerate(a):
        b.append(2/item)

except Exception:
    b.append(0)
    iterate_append(a[i+1:])

print(b)

=======
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

>>>>>>> 4b5592f1f54e4882ed2d8e819ccb7d83623b391c
