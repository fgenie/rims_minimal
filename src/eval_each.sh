EXEPATH=/Users/seonils/dev/rims_minimal/src/
cd $EXEPATH

echo "#chatGPT RESULTS"
echo "##GSM"
python run_evaluation_each.py --eval_jslf /Users/seonils/dev/rims_minimal/src/seonil_scripts/0_RESULTS/gsm/chatgpt_01_25_02_00_chatgpt_gsm8K_test_01_25_00_45_model_selection3_startidx0_rims_startidx0.jsonl --eval_type gsm 
echo "##SVAMP"
python run_evaluation_each.py --eval_jslf /Users/seonils/dev/rims_minimal/src/seonil_scripts/0_RESULTS/svamp/chatgpt_01_25_02_42_chatgpt_svamp_01_25_02_12_model_selection3_startidx0_rims_startidx0.jsonl --eval_type svamp
echo "##MATH"
python run_evaluation_each.py --eval_jslf /Users/seonils/dev/rims_minimal/src/seonil_scripts/0_RESULTS/math/baseline_chatgpt.jsonl --eval_type math 

echo "=============="
echo "#GPT4TURBO RESULTS"
echo "##MATH"
python run_evaluation_each.py --eval_jslf /Users/seonils/dev/rims_minimal/src/seonil_scripts/0_RESULTS/math/rims_gpt4turbo.jsonl --eval_type math 

