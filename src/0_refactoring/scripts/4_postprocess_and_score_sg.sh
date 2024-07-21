
# F1=outputs/gsm8K_test_dt.gsm/Meta-Llama-3-8B-Instruct/simple_greedy/n1_0.0_sg_raw_query_result.jsonl
# F2=outputs/ocw_course_dt.ocw/Meta-Llama-3-8B-Instruct/simple_greedy/n1_0.0_sg_raw_query_result.jsonl
# F3=outputs/MATH-full_dt.math/Meta-Llama-3-8B-Instruct/simple_greedy/n1_0.0_sg_raw_query_result.jsonl

# for F in $F1 $F2 $F3; do
#     python postprocess_rawouts.py process_simple_greedy --infile $F
# done



F1=outputs/gsm8K_test_dt.gsm/Meta-Llama-3-8B-Instruct/simple_greedy/processed_sg.jsonl
F2=outputs/ocw_course_dt.ocw/Meta-Llama-3-8B-Instruct/simple_greedy/processed_sg.jsonl
F3=outputs/MATH-full_dt.math/Meta-Llama-3-8B-Instruct/simple_greedy/processed_sg.jsonl
for F in $F1 $F2 $F3; do
    python score_processed.py score_sg --ptn $F
done
