

PTN=outputs/*/Meta-Llama-3-8B-Instruct/rims/rims*/n1_0.0_rims_raw_query_result.jsonl

python postprocess_rawouts.py process_rims --infile $PTN

PTN1=outputs/*/Meta-Llama-3-8B-Instruct/rims/rims*/processed_rims.jsonl

python score_processed.py score_rims --ptn $PTN1
