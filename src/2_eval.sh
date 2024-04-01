
set -x

EXEPATH=/Users/seonils/dev/rims_minimal/PREV_ocw=
cd $EXEPATH

GSM_INFERRED=outputs/gsm8K_test_dt.gsm/chatgpt0613long/model_selection_prompts/03_25_01_21_45.jsonl
OCW_INFERRED=outputs/ocw_course_dt.ocw/chatgpt0613long/model_selection_prompts/03_23_15_38_01.jsonl
MATH_INFERRED=outputs/MATH-full_dt.math/chatgpt0613long/model_selection_prompts/merged.jsonl


# RIMS/BASELINE PATHS here
# outputs/gsm8K_test_dt.gsm/chatgpt0613long/model_selection_prompts/03_23_14_35_13.jsonl
GSM_OLD_RIMS_GSMPROMPT=outputs/gsm8K_test_dt.gsm/chatgpt0613long/rims_gsm_best.txt/03_27_00_26_05.jsonl
GSM_NEW_RIMS=outputs/gsm8K_test_dt.gsm/chatgpt0613long/rims_gsm_best_newer.txt/03_27_00_33_57.jsonl
GSM_NEW_RIMS_H=outputs/gsm8K_test_dt.gsm/chatgpt0613long/rims_gsm_best_newer-hint.txt/03_27_00_41_56.jsonl
GSM_NEW_RIMS_HM=outputs/gsm8K_test_dt.gsm/chatgpt0613long/rims_gsm_best_newer-hint-mistakes.txt/03_27_00_48_31.jsonl
GSM_NEW_RIMS_HMA=outputs/gsm8K_test_dt.gsm/chatgpt0613long/rims_gsm_best_newer-hint-mistakes-attempt1.txt/03_27_00_54_03.jsonl

# outputs/ocw_course_dt.ocw/chatgpt0613long/model_selection_prompts/03_23_15_38_01.jsonl # turned out bad (ocw_results.txt)
OCW_OLD_RIMS_GSMPROMPT=outputs/ocw_course_dt.ocw/chatgpt0613long/rims_gsm_best.txt/03_24_03_09_20.jsonl
OCW_RIMS=outputs/ocw_course_dt.ocw/chatgpt0613long/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt/03_24_03_15_14.jsonl
OCW_RIMS_H=outputs/ocw_course_dt.ocw/chatgpt0613long/rims_ocw_p2c-cot.pal-p2c.pal-cot_-hint.txt/03_24_03_22_40.jsonl
OCW_RIMS_HM=outputs/ocw_course_dt.ocw/chatgpt0613long/rims_ocw_p2c-cot.pal-p2c.pal-cot_-hint-mistakes.txt/03_24_03_29_30.jsonl
OCW_RIMS_HMA=outputs/ocw_course_dt.ocw/chatgpt0613long/rims_ocw_p2c-cot.pal-p2c.pal-cot_-hint-mistakes-attempt1.txt/03_24_03_36_05.jsonl


# new ocw prompts (replaced pal-cot w/ cot-p2c)
OCW_RIMS_1=outputs/ocw_course_dt.ocw/chatgpt0613long/rims_ocw_p2c-cot.pal-p2c.cot-p2c__.txt/03_27_01_00_31.jsonl
OCW_RIMS_H_1=outputs/ocw_course_dt.ocw/chatgpt0613long/rims_ocw_p2c-cot.pal-p2c.cot-p2c_-hint.txt/03_27_01_07_51.jsonl
OCW_RIMS_HM_1=outputs/ocw_course_dt.ocw/chatgpt0613long/rims_ocw_p2c-cot.pal-p2c.cot-p2c_-hint-mistakes.txt/03_27_01_14_48.jsonl
OCW_RIMS_HMA_1=outputs/ocw_course_dt.ocw/chatgpt0613long/rims_ocw_p2c-cot.pal-p2c.cot-p2c_-hint-mistakes-attempt1.txt/03_27_01_21_48.jsonl

#math results 1
MATH_OLD_RIMS_GSMPROMT=outputs/MATH-full_dt.math/chatgpt0613long/rims_gsm_best.txt/03_25_02_13_27.jsonl

MATH_RIMS_1=outputs/MATH-full_dt.math/chatgpt0613long/rims_math_p2c-cot.pal-p2c.pal-cot__.txt1/03_25_08_19_30.jsonl
MATH_RIMS_1_H=outputs/MATH-full_dt.math/chatgpt0613long/rims_math_p2c-cot.pal-p2c.pal-cot_-hint.txt1/03_25_09_55_12.jsonl
MATH_RIMS_1_HM=outputs/MATH-full_dt.math/chatgpt0613long/rims_math_p2c-cot.pal-p2c.pal-cot_-hint-mistakes.txt1/03_25_11_30_44.jsonl
MATH_RIMS_1_HMA=outputs/MATH-full_dt.math/chatgpt0613long/rims_math_p2c-cot.pal-p2c.pal-cot_-hint-mistakes-attempt1.txt1/03_25_13_09_14.jsonl

MATH_RIMS=outputs/MATH-full_dt.math/chatgpt0613long/rims_math_p2c-cot.pal-p2c.pal-cot__.txt/03_25_23_39_28.jsonl
MATH_RIMS_H=outputs/MATH-full_dt.math/chatgpt0613long/rims_math_p2c-cot.pal-p2c.pal-cot_-hint.txt/03_25_04_25_11.jsonl
MATH_RIMS_HM=outputs/MATH-full_dt.math/chatgpt0613long/rims_math_p2c-cot.pal-p2c.pal-cot_-hint-mistakes.txt/03_25_06_14_18.jsonl
MATH_RIMS_HMA=outputs/MATH-full_dt.math/chatgpt0613long/rims_math_p2c-cot.pal-p2c.pal-cot_-hint-mistakes-attempt1.txt/03_26_02_04_23.jsonl



# python run_evaluation_new.py --eval_jslf $GSM_INFERRED --eval_type gsm --outf gsm_baseline.txt --eval_indiv_and_overlap
# python run_evaluation_new.py --eval_jslf $OCW_INFERRED --eval_type ocw --outf ocw_baseline.txt --eval_indiv_and_overlap
# python run_evaluation_new.py --eval_jslf $MATH_INFERRED --eval_type math --outf math_baseline.txt --eval_indiv_and_overlap



# # # commands here with for loop maybe

# for RES in $GSM_OLD_RIMS_GSMPROMPT $GSM_NEW_RIMS $GSM_NEW_RIMS_H $GSM_NEW_RIMS_HM $GSM_NEW_RIMS_HMA
# do
#     python run_evaluation_new.py \
#         --eval_type gsm \
#         --eval_jslf ${RES} --outf gsm_results.txt
# done

# for RES in $OCW_RIMS_1 $OCW_RIMS_H_1 $OCW_RIMS_HM_1 $OCW_RIMS_HMA_1
# do
#     python run_evaluation_new.py \
#         --eval_type ocw \
#         --eval_jslf ${RES} --outf ocw_results_new.txt
# done



# for RES in $OCW_OLD_RIMS_GSMPROMPT $OCW_RIMS $OCW_RIMS_H $OCW_RIMS_HM $OCW_RIMS_HMA
# do
#     python run_evaluation_new.py \
#         --eval_type ocw \
#         --eval_jslf ${RES} --outf ocw_results.txt
# done



# for res in $MATH_OLD_RIMS_GSMPROMT $MATH_RIMS_1 $MATH_RIMS_1_H $MATH_RIMS_1_HM $MATH_RIMS_1_HMA $MATH_RIMS 
# for res in $MATH_RIMS_H $MATH_RIMS_HM $MATH_RIMS_HMA
# do
#     python run_evaluation_new.py \
#         --eval_type math \
#         --eval_jslf ${res} --outf math_results.txt
# done




## ================================================== ###

## ================================================== ###
GSM_INFERRED1106=outputs/gsm8K_test_dt.gsm/chatgpt1106/model_selection_prompts/03_30_02_18_36.jsonl
OCW_INFERRED1106=outputs/ocw_course_dt.ocw/chatgpt1106/model_selection_prompts/03_30_01_27_09.jsonl
# MATH_INFERRED1106=outputs/MATH-full_dt.math/chatgpt1106



GSM_OLD_RIMS_GSMPROMPT_1106=outputs/gsm8K_test_dt.gsm/chatgpt1106/rims_gsm_best.txt/03_30_11_54_29.jsonl
GSM_NEW_RIMS_1106=outputs/gsm8K_test_dt.gsm/chatgpt1106/rims_gsm_best_newer.txt/src/outputs/gsm8K_test_dt.gsm/chatgpt1106/rims_gsm_best_newer.txt/03_30_12_08_44.jsonl
GSM_NEW_RIMS_H_1106=outputs/gsm8K_test_dt.gsm/chatgpt1106/rims_gsm_best_newer-hint.txt/03_30_12_22_20.jsonl
GSM_NEW_RIMS_HM_1106=outputs/gsm8K_test_dt.gsm/chatgpt1106/rims_gsm_best_newer-hint-mistakes.txt/03_30_12_36_29.jsonl
GSM_NEW_RIMS_HMA_1106=outputs/gsm8K_test_dt.gsm/chatgpt1106/rims_gsm_best_newer-hint-mistakes-attempt1.txt/03_30_12_48_40.jsonl





OCW_RIMS_1_1106=outputs/ocw_course_dt.ocw/chatgpt1106/rims_ocw_p2c-cot.pal-p2c.cot-p2c__.txt/03_30_09_14_06.jsonl
OCW_RIMS_H_1_1106=outputs/ocw_course_dt.ocw/chatgpt1106/rims_ocw_p2c-cot.pal-p2c.cot-p2c_-hint.txt/03_30_09_24_29.jsonl
OCW_RIMS_HM_1_1106=outputs/ocw_course_dt.ocw/chatgpt1106/rims_ocw_p2c-cot.pal-p2c.cot-p2c_-hint-mistakes.txt/03_30_09_34_02.jsonl
OCW_RIMS_HMA_1_1106=outputs/ocw_course_dt.ocw/chatgpt1106/rims_ocw_p2c-cot.pal-p2c.cot-p2c_-hint-mistakes-attempt1.txt/03_30_09_43_07.jsonl



# new ocw prompts (replaced pal-cot w/ cot-p2c)
OCW_OLD_RIMS_GSMPROMPT_1106=outputs/ocw_course_dt.ocw/chatgpt1106/rims_gsm_best.txt/03_30_11_07_00.jsonl
OCW_RIMS_1106=outputs/ocw_course_dt.ocw/chatgpt1106/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt/03_30_11_17_44.jsonl
OCW_RIMS_H_1106=outputs/ocw_course_dt.ocw/chatgpt1106/rims_ocw_p2c-cot.pal-p2c.pal-cot_-hint.txt/03_30_11_27_49.jsonl
OCW_RIMS_HM_1106=outputs/ocw_course_dt.ocw/chatgpt1106/rims_ocw_p2c-cot.pal-p2c.pal-cot_-hint-mistakes.txt/03_30_11_37_51.jsonl
OCW_RIMS_HMA_1106=outputs/ocw_course_dt.ocw/chatgpt1106/rims_ocw_p2c-cot.pal-p2c.pal-cot_-hint-mistakes-attempt1.txt/03_30_11_46_57.jsonl










# python run_evaluation_new.py --eval_jslf $GSM_INFERRED1106 --eval_type gsm --outf gsm_baseline1106.txt --eval_indiv_and_overlap
# python run_evaluation_new.py --eval_jslf $OCW_INFERRED1106 --eval_type ocw --outf ocw_baseline1106.txt --eval_indiv_and_overlap





# for RES in $OCW_RIMS_1_1106 $OCW_RIMS_H_1_1106 $OCW_RIMS_HM_1_1106 $OCW_RIMS_HMA_1_1106
# do
#     python run_evaluation_new.py \
#         --eval_type ocw \
#         --eval_jslf ${RES} --outf ocw_results_new1106_1.txt
# done




for RES in $GSM_OLD_RIMS_GSMPROMPT_1106 $GSM_NEW_RIMS_1106 $GSM_NEW_RIMS_H_1106 $GSM_NEW_RIMS_HM_1106 $GSM_NEW_RIMS_HMA_1106
do
    python run_evaluation_new.py \
        --eval_type gsm \
        --eval_jslf ${RES} --outf outputs/gsm_results1106.txt
done


python run_evaluation_new.py --eval_jslf /Users/seonils/dev/rims_minimal/src/outputs/gsm8K_test_dt.gsm/chatgpt1106/rims_gsm_best_newer.txt/03_30_12_08_44.jsonl --eval_type gsm --outf outputs/gsm_results1106.txt

# for RES in $OCW_OLD_RIMS_GSMPROMPT $OCW_RIMS $OCW_RIMS_H $OCW_RIMS_HM $OCW_RIMS_HMA
# do
#     python run_evaluation_new.py \
#         --eval_type ocw \
#         --eval_jslf ${RES} --outf ocw_results_new_1106_1.txt
# done




## old prompt overlaps 
# PREV_ocw=seonil_scripts/0_RES_v3/ocw_0613long/chatgpt0613long_rims_ocw.jsonl
# PREV_gsm=seonil_scripts/0_RES_v3/gsm_0613long/chatgpt0613long_rims_gsm.jsonl
# PREV_math=seonil_scripts/0_RES_v3/math_full_0613long/chatgpt0613long_rims_mathmerged.jsonl
# python run_evaluation_new.py --eval_jslf $PREV_ocw --eval_type gsm --outf old_ocw.txt --eval_indiv_and_overlap
# python run_evaluation_new.py --eval_jslf $PREV_gsm --eval_type ocw --outf old_gsm.txt --eval_indiv_and_overlap
# python run_evaluation_new.py --eval_jslf $PREV_math --eval_type math --outf old_math.txt --eval_indiv_and_overlap




# python run_evaluation_new.py --eval_jslf $MATH_INFERRED1106 --eval_type math --outf math_baseline.txt --eval_indiv_and_overlap