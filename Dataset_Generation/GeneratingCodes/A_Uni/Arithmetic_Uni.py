import random
import Arithmetic_Uni_1_Structure_Gen
import Arithmetic_Uni_2_Conversation_Gen


random.seed(42)

def main():

    Arithmetic_Uni_1_Structure_Gen.generate_dataset()
    Arithmetic_Uni_2_Conversation_Gen.main()
if __name__ == "__main__":
    main()