
F1=outputs/gsm8K_test_dt.gsm/Meta-Llama-3-8B-Instruct/processed_indiv.jsonl
F2=outputs/ocw_course_dt.ocw/Meta-Llama-3-8B-Instruct/processed_indiv.jsonl
F3=outputs/MATH-full_dt.math/Meta-Llama-3-8B-Instruct/processed_indiv.jsonl
for F in $F1 $F2 $F3; do
    python run_sg.py --indiv_processed_jslf $F
done
