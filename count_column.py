
header = input("Enter the header: ")

# iterate and count commas
count = 0
for c in header:
    if c == ',':
        count += 1

print(f"Number of columns: {count + 1}")