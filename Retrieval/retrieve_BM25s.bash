
echo "----------------------------  BM25  ----------------------------------"
echo "Retrieving for track A, type Multi, retriever_name bm25"
python ./Retrieval/retrieve_indexing.py --output_folder ./Retrieval/Results/ --track A --type Multi --retriever_name bm25


echo "Retrieving for track A, type Uni, retriever_name bm25"
python ./Retrieval/retrieve_indexing.py --output_folder ./Retrieval/Results/ --track A --type Uni --retriever_name bm25


echo "Retrieving for track T, type Multi, retriever_name bm25"
python ./Retrieval/retrieve_indexing.py --output_folder ./Retrieval/Results/ --track T --type Multi --retriever_name bm25


echo "Retrieving for track T, type Uni, retriever_name bm25"
python ./Retrieval/retrieve_indexing.py --output_folder ./Retrieval/Results/ --track T --type Uni --retriever_name bm25


echo "Retrieving for track W, type Multi, retriever_name bm25"
python ./Retrieval/retrieve_indexing.py --output_folder ./Retrieval/Results/ --track W --type Multi --retriever_name bm25


echo "Retrieving for track W, type Uni, retriever_name bm25"
python ./Retrieval/retrieve_indexing.py --output_folder ./Retrieval/Results/ --track W --type Uni --retriever_name bm25





