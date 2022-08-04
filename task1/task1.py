def multiplicate(A):
    multiply = 1
    multiply_non_zero = 1
    counter_zero = 0
    for items in A:
        multiply = multiply * items
        if items == 0:
            counter_zero += 1
        else:
            multiply_non_zero *= items
            
    output = [0] * len(A)
    if multiply != 0:
        for i in range(len(A)):
            output[i] = multiply // A[i]
    elif counter_zero == 1:
        for i in range(len(A)):
            if A[i] == 0:
                output[i] = multiply_non_zero
    return output



input = [1,2,3,4]
output = multiplicate(input)
print(output)
            
