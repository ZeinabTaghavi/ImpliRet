import random
import Arithmetic_Multi_1_Structure_Gen
import Arithmetic_Multi_2_Conversation_Gen


random.seed(42)

def main():

    Arithmetic_Multi_1_Structure_Gen.generate_dataset()
    Arithmetic_Multi_2_Conversation_Gen.main()
if __name__ == "__main__":
    main()