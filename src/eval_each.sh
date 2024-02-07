EXEPATH=/Users/seonils/dev/rims_minimal/src/
cd $EXEPATH

echo "# 1106 chatGPT RESULTS" >> result.txt
echo "## GSM" >> result.txt
python run_evaluation_each.py --eval_jslf seonil_scripts/0_RESULTS_v1/gsm/chatgpt_model_selection3_gsm.jsonl --eval_type gsm 
echo "## SVAMP" >> result.txt
python run_evaluation_each.py --eval_jslf seonil_scripts/0_RESULTS_v1/svamp/chatgpt_model_selection3_svamp.jsonl --eval_type svamp
echo "## OCW" >> result.txt
python run_evaluation_each.py --eval_jslf seonil_scripts/0_RESULTS_v1/ocw/chatgpt_model_selection3_ocw.jsonl --eval_type ocw
