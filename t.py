def iterate_append(test_list):

    try:
        for i, item in enumerate(test_list):
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

