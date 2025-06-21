
echo "----------------------------  BM25  ----------------------------------"
echo "Retrieving for category arithmetic, discourse_type multispeaker, retriever_name bm25"
python ./Retrieval/retrieve_indexing.py --output_folder ./Retrieval/results/ --category arithmetic --discourse_type multispeaker --retriever_name bm25


echo "Retrieving for category arithmetic, discourse_type unispeaker, retriever_name bm25"
python ./Retrieval/retrieve_indexing.py --output_folder ./Retrieval/results/ --category arithmetic --discourse_type unispeaker --retriever_name bm25


echo "Retrieving for category temporal, discourse_type multispeaker, retriever_name bm25"
python ./Retrieval/retrieve_indexing.py --output_folder ./Retrieval/results/ --category temporal --discourse_type multispeaker --retriever_name bm25


echo "Retrieving for category temporal, discourse_type unispeaker, retriever_name bm25"
python ./Retrieval/retrieve_indexing.py --output_folder ./Retrieval/results/ --category temporal --discourse_type unispeaker --retriever_name bm25


echo "Retrieving for category wknowledge, discourse_type multispeaker, retriever_name bm25"
python ./Retrieval/retrieve_indexing.py --output_folder ./Retrieval/results/ --category wknowledge --discourse_type multispeaker --retriever_name bm25


echo "Retrieving for category wknowledge, discourse_type unispeaker, retriever_name bm25"
python ./Retrieval/retrieve_indexing.py --output_folder ./Retrieval/results/ --category wknowledge --discourse_type unispeaker --retriever_name bm25





