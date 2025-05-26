import sys
import pandas

def main():
    # Expected 5 arguments + 1 for script name
    if len(sys.argv) != 6:
        print("Usage: python code.py <nParticles> <density> <nInitCycle> <nCycle> <nSpacing>")
        sys.exit(1)

    errors = []

    # Individual parsing with detailed error tracking
    try:
        nParticles = int(sys.argv[1])
    except ValueError:
        errors.append(f"Invalid nParticles: expected an integer, got '{sys.argv[1]}'.")

    try:
        density = float(sys.argv[2])
    except ValueError:
        errors.append(f"Invalid density: expected a float, got '{sys.argv[2]}'.")

    try:
        nInitCycle = int(sys.argv[3])
    except ValueError:
        errors.append(f"Invalid nInitCycle: expected an integer, got '{sys.argv[3]}'.")

    try:
        nCycle = int(sys.argv[4])
    except ValueError:
        errors.append(f"Invalid nCycle: expected an integer, got '{sys.argv[4]}'.")

    try:
        nSpacing = int(sys.argv[5])
    except ValueError:
        errors.append(f"Invalid nSpacing: expected an integer, got '{sys.argv[5]}'.")

    # If there were any parsing errors, print and exit
    if errors:
        print("\nErrors detected:")
        for error in errors:
            print(" -", error)
        sys.exit(1)

    # Output the collected variables
    print("\nCollected Inputs:")
    print(f"nParticles: {nParticles}")
    print(f"density: {density}")
    print(f"nInitCycle: {nInitCycle}")
    print(f"nCycle: {nCycle}")
    print(f"nSpacing: {nSpacing}")

    pm, pu, em, eu, acc_rate, pressure, energy = MC_NVT(nParticles, density, nInitCycle, nCycle, nSpacing)

    # create a pandas dataframe
    # save the dataframe as a file


if __name__ == "__main__":
    main()