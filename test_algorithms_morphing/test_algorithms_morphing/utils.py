def algorithm_name(algorithm: int) -> str:
    if algorithm == 1:
        return "RFC"
    elif algorithm == 2:
        return "MLP"
    elif algorithm == 3:
        return "KNN"
    elif algorithm == 4:
        return "Logistic"
    else:
        return "Unknown"

def choose_algorithm() -> int:
    print("Available algorithms:")
    print("1. Random Forest Classifier")
    print("2. MLP Classifier")
    print("3. KNN Classifier")
    print("4. Logistic Regression")
    algorithm1 = int(input("Choose the first algorithm (enter the corresponding number): "))
    algorithm2 = int(input("Choose the second algorithm (enter the corresponding number): "))
    return algorithm1, algorithm2