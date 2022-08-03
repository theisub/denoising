
input = [1,2,3,4]
multiply = 1
multiply_non_zero = 1
counter_zero = 0
for items in input:
    multiply = multiply * items
    if items == 0:
        counter_zero += 1
    else:
        multiply_non_zero *= items
        
output = [0] * len(input)
if multiply != 0:
    for i in range(len(input)):
        output[i] = multiply // input[i]
elif counter_zero == 1:
    for i in range(len(input)):
        if input[i] == 0:
            output[i] = multiply_non_zero

print(output)
            
