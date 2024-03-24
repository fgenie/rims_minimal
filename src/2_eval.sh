
GSM_INFERRED=outputs/gsm8K_test_dt.gsm/chatgpt0613long/model_selection_prompts/03_23_14_35_13.jsonl
OCW_INFERRED=outputs/ocw_course_dt.ocw/chatgpt0613long/model_selection_prompts/03_23_15_38_01.jsonl

# RIMS/BASELINE PATHS here
# outputs/gsm8K_test_dt.gsm/chatgpt0613long/model_selection_prompts/03_23_14_35_13.jsonl
GSM_OLD_RIMS_GSMPROMPT=outputs/gsm8K_test_dt.gsm/chatgpt0613long/rims_gsm_best.txt/03_24_02_59_41.jsonl
GSM_NEW_RIMS=outputs/gsm8K_test_dt.gsm/chatgpt0613long/rims_gsm_best_newer.txt/03_23_23_48_40.jsonl
GSM_NEW_RIMS_H=outputs/gsm8K_test_dt.gsm/chatgpt0613long/rims_gsm_best_newer-hint.txt/03_24_00_00_50.jsonl
GSM_NEW_RIMS_HM=outputs/gsm8K_test_dt.gsm/chatgpt0613long/rims_gsm_best_newer-hint-mistakes.txt/03_24_00_11_40.jsonl
GSM_NEW_RIMS_HMA=outputs/gsm8K_test_dt.gsm/chatgpt0613long/rims_gsm_best_newer-hint-mistakes-attempt1.txt/03_24_00_20_51.jsonl

# outputs/ocw_course_dt.ocw/chatgpt0613long/model_selection_prompts/03_23_15_38_01.jsonl
OCW_OLD_RIMS_GSMPROMPT=outputs/ocw_course_dt.ocw/chatgpt0613long/rims_gsm_best.txt/03_24_03_09_20.jsonl
OCW_RIMS=outputs/ocw_course_dt.ocw/chatgpt0613long/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt/03_24_03_15_14.jsonl
OCW_RIMS_H=outputs/ocw_course_dt.ocw/chatgpt0613long/rims_ocw_p2c-cot.pal-p2c.pal-cot_-hint.txt/03_24_03_22_40.jsonl
OCW_RIMS_HM=outputs/ocw_course_dt.ocw/chatgpt0613long/rims_ocw_p2c-cot.pal-p2c.pal-cot_-hint-mistakes.txt/03_24_03_29_30.jsonl
OCW_RIMS_HMA=outputs/ocw_course_dt.ocw/chatgpt0613long/rims_ocw_p2c-cot.pal-p2c.pal-cot_-hint-mistakes-attempt1.txt/03_24_03_36_05.jsonl



python run_evaluation_new.py --eval_jslf $GSM_INFERRED --eval_type gsm --outf gsm_baseline.txt --eval_indiv_and_overlap
python run_evaluation_new.py --eval_jslf $OCW_INFERRED --eval_type ocw --outf ocw_baseline.txt --eval_indiv_and_overlap

# commands here with for loop maybe

for RES in $GSM_OLD_RIMS_GSMPROMPT $GSM_NEW_RIMS $GSM_NEW_RIMS_H $GSM_NEW_RIMS_HM $GSM_NEW_RIMS_HMA
do
    python run_evaluation_new.py \
        --eval_type gsm \
        --eval_jslf ${RES} --outf gsm_results.txt
done



for RES in $OCW_OLD_RIMS_GSMPROMPT $OCW_RIMS $OCW_RIMS_H $OCW_RIMS_HM $OCW_RIMS_HMA
do
    python run_evaluation_new.py \
        --eval_type ocw \
        --eval_jslf ${RES} --outf ocw_results.txt
done



