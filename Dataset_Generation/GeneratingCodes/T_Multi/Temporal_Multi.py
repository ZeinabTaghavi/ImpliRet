import random
import Temporal_Multi_1_Structure_Gen
import Temporal_Multi_2_Conversation_Gen


random.seed(42)

def main():

    Temporal_Multi_1_Structure_Gen.generate_dataset()
    Temporal_Multi_2_Conversation_Gen.main()
if __name__ == "__main__":
    main()