def main():
    try:
        f = open("theta_values.txt", "r")
        t0 = float(f.readline().strip())
        t1 = float(f.readline().strip())

        miles = float(input("Give the mileage of the car: "))
        price = t0 + t1 * miles
        print("Estimated price for the car: ", str(price))
    except Exception as e:
        print("Error: ", e)

main()
