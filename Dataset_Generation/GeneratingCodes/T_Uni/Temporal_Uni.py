import random
import Temporal_Uni_1_Structure_Gen
import Temporal_Uni_2_Conversation_Gen


random.seed(42)

def main():

    Temporal_Uni_1_Structure_Gen.generate_dataset()
    Temporal_Uni_2_Conversation_Gen.main()
if __name__ == "__main__":
    main()