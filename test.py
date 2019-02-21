a = [1,2,3,0,4]
test_b = []
def enumerate_append(remaining_list, b):

    for i, num in enumerate(remaining_list):
        try:
            b.append(4./num)

        except Exception:
            b.append(0)
            print("error happens to {}".format(num))
            enumerate_append(remaining_list[i+1:], b)


enumerate_append(a, test_b)

print(test_b)